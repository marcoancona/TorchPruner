import torch
import numpy as np
import torch.nn as nn

def attributions_for_module(self, module, method):
    self.model.eval()
    n = module.weight.shape[0]

    if method.startswith("occlusion"):
        raise RuntimeError("Occlusion not valid")
        # return _occlusion(self, module)
    elif method.startswith("sv"):
        return _fast_estimate_sv_for_module(self, module, method)
    elif method.startswith("intgrad"):
        raise RuntimeError("Intgrad not valid")
        # return integrated_gradients(self, module, method)
    elif method.startswith("weight"):
        return _sum_absolute_weights(module)
    elif method.startswith("random"):
        return np.random.random((n,))

    self.output = None
    activations_count = torch.zeros((n,)).to(self.device)
    gradients = torch.zeros((n,)).to(self.device)
    gradients_x_input_abs = torch.zeros((n,)).to(self.device)
    gradients_x_input = torch.zeros((n,)).to(self.device)

    def _forward_hook(module, input, output):
        self.output = output
        _output = (output > 0).float()
        while len(_output.shape) >= 3:
            _output = _output.sum(-1)
        _output = _output.sum(0)
        activations_count.add_(_output)

    def _backward_hook(module, grad_input, grad_output):
        _gradients = grad_output[0].abs()
        if len(_gradients.shape) >= 3:
            _gradients = _gradients.flatten(2).sum(-1)

        _gradients = _gradients.sum(0)
        gradients.add_(_gradients)

        _gradients_x_input = grad_output[0] * self.output
        if len(_gradients_x_input.shape) >= 3:
            _gradients_x_input = _gradients_x_input.flatten(2).mean(-1)

        gradients_x_input.add_(-1. * _gradients_x_input.sum(0))
        gradients_x_input_abs.add_(_gradients_x_input.abs().sum(0))

    found = False
    for module_name, m in self.model.named_modules():
        if module == m:
            found = True
        elif isinstance(m, nn.ReLU) and found is True:
            module = m
            break

    h1 = module.register_forward_hook(_forward_hook)
    h2 = module.register_backward_hook(_backward_hook)

    n_samples = len(self.data_loader.dataset)
    print ("N Samples: ", n_samples)

    for batch_idx, (data, target) in enumerate(self.data_loader):
        data, target = data.to(self.device), target.to(self.device)

        _, _, loss = self._run_forward(x=data, y_true=target,)
        loss.backward()

    h1.remove()
    h2.remove()

    if method.startswith("grad"):
        return (gradients / n_samples).detach().cpu().numpy()
    elif method.startswith("taylor"):
        if "abs" in method:
            return (gradients_x_input_abs / n_samples).detach().cpu().numpy()
        else:
            return (gradients_x_input / n_samples).detach().cpu().numpy()
    elif method.startswith("count"):
        return (activations_count / n_samples).detach().cpu().numpy()


def _sum_absolute_weights(module):
    w = module.weight.detach().cpu().numpy()
    w = np.abs(w)
    while len(w.shape) > 1:
        w = w.sum(-1)
    return w


#


def _fast_estimate_sv_for_module(self, module, method, sv_samples=10):
    # Estimate Shapley Values of module's output nodes

    # Number of prunable units
    n = module.weight.shape[0]

    # Lets find the next ReLU. TODO: is it necessary? What about if BatchNorm is in between?
    found = False
    for module_name, m in self.model.named_modules():
        if module == m:
            found = True
        elif isinstance(m , nn.ReLU) and found is True:
            module = m
            break

    # Start sampling

    with torch.no_grad():
        # Get number of samples
        if "#" in method:
            import re
            sv_samples = int(re.findall(r"\d+", method.split("#")[1])[0])

        print("SV, samples ", sv_samples)

        sv = torch.zeros((len(self.data_loader.dataset), sv_samples, n,)).to(self.device)
        permutations = [np.random.permutation(n) for _ in range(sv_samples)]

        for batch_idx, (data, target) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            # print(f"{batch_idx}")

            # Compute activations of current module
            activations, _, __ = self._run_forward(
                data, return_intermediate_output_module=module, reduction="none"
            )

            # Compute accuracy with no players
            _, full_accuracy, full_loss = self._run_forward(
                x=activations, y_true=target, process_as_intermediate_output_module=module, reduction="none",
            )
            _, no_player_accuracy, no_player_loss = self._run_forward(
                x=torch.zeros_like(activations),
                y_true=target,
                process_as_intermediate_output_module=module, reduction="none",
            )

            # print("Loss gap", (no_player_loss - full_loss).mean())

            for j in range(sv_samples):

                _activation = activations.clone().detach()
                loss = full_loss.clone().detach()
                accuracy = full_accuracy

                for i in permutations[j]: #[:min(n, int(0.1*n))]:
                    _activation.index_fill_(
                        1, torch.tensor(np.array([i])).long().to(self.device), 0.0
                    )
                    _, new_accuracy, new_loss = self._run_forward(
                        x=_activation,
                        y_true=target,
                        process_as_intermediate_output_module=module,reduction="none"
                    )

                    if "loss" in method:
                        # delta: B x 1
                        delta = new_loss - loss
                        n = delta.shape[0]
                        sv[batch_idx*n:(batch_idx+1)*n, j, i] += delta
                    elif "acc" in method:
                        sv[:, j, i] += accuracy - new_accuracy
                    else:
                        raise RuntimeError(f"Should be sv-loss or sv-acc, not {method}")
                    # count[i] += 1.0
                    loss = new_loss
                    accuracy = new_accuracy

        # Once sampling is completed, test consistency among input samples and reduce SV to a single aggregated value
        # SV: B x SAMPLES x N

        try:
            import h5py
            f = h5py.File('sv.hdf5', 'a')
            path = f"{self.experiment_id}/{self._module_name(module)}/{self._epoch}/sv"
            if path in f:
                del f[path]
            f.create_dataset(path, data=sv.detach().cpu().numpy())
            f.close()
        except:
            print ("Warn: could not open sv.h5")

        if "abs" in method:
            sv = sv.abs()

        # mean over samples, as Shapley formula
        sv = sv.mean(1)

        # aggregate over games
        if "-97p" in method:
            sv = torch.kthvalue(sv, int(sv.shape[0] * 0.977), dim=0)[0]
        elif "-99p" in method:
            sv = torch.kthvalue(sv, int(sv.shape[0] * 0.995), dim=0)[0]
        elif "-2std" in method:
            sv = sv.mean(0) + 2 * sv.std(0)
        elif "-borda" in method:
            sv = sv.argsort(1).float().mean(0)
        else:
            sv = sv.mean(0)

        if self.verbose > 0:
            # assert not np.isnan(sv).any(), "nans!!"
            print(f"Estimating Shapley Values for {n} players in {module}")
            print (f"SV shape: {sv.shape}")
            print(f"Shapley Values sum to {sv.sum()}")
            print(f"This should match the loss gap {(no_player_loss - full_loss).mean()}")
            print(f"... or the accuracy gap {full_accuracy - no_player_accuracy}")

        return sv.detach().cpu().numpy()


# def _occlusion(self, module, sv_samples=1):
#     # Estimate Shapley Values of module's output nodes
#     n = module.weight.shape[0]
#
#     with torch.no_grad():
#         data, target = next(iter(self.data_loader))
#         data, target = data.to(self.device), target.to(self.device)
#
#         # Compute activations of current module
#         activations, _, __ = self._run_forward(
#             data, return_intermediate_output_module=module
#         )
#
#         # Compute accuracy with no players
#         _, full_accuracy, full_loss = self._run_forward(
#             x=activations, y_true=target, process_as_intermediate_output_module=module,
#         )
#         _, no_player_accuracy, no_player_loss = self._run_forward(
#             x=torch.zeros_like(activations),
#             y_true=target,
#             process_as_intermediate_output_module=module,
#         )
#
#         # Start SV sampling
#         sv = torch.zeros((n,)).float().to(self.device)
#         count = torch.zeros((n,)).float().to(self.device)
#
#         for j in range(1):
#             # if self.verbose > 0:
#             #     print(f"SV - Iteration {j}")
#             _activation = activations.clone().detach()
#             _, accuracy, loss = self._run_forward(
#                 x=_activation,
#                 y_true=target,
#                 process_as_intermediate_output_module=module,
#             )
#             for i in np.random.permutation(n):
#                 a = _activation.index_fill(
#                     1, torch.tensor(np.array([i])).long().to(self.device), 0.0
#                 )
#                 _, new_accuracy, new_loss = self._run_forward(
#                     x=a, y_true=target, process_as_intermediate_output_module=module,
#                 )
#                 sv[i] += new_loss - loss  # .clone().detach()
#                 # sv[i] += (accuracy - new_accuracy)
#                 count[i] += 1.0
#                 # loss = new_loss
#                 accuracy = new_accuracy
#         sv /= sv_samples
#
#         if self.verbose > 0:
#             print(f"Estimating Occlusion for {n} players in {module}")
#
#     return sv.detach().cpu().numpy()

# def integrated_gradients(self, module, method, sv_samples=10):
#     # Estimate Shapley Values of module's output nodes
#     n = module.weight.shape[0]
#
#     sv = None # np.zeros((n,))
#     count = np.zeros((n,))  + 0.0
#
#     found = False
#     for module_name, m in self.model.named_modules():
#         if module == m:
#             found = True
#         elif isinstance(m , nn.ReLU) and found is True:
#             module = m
#             break
#
#
#
#     # Start SV sampling
#
#
#     if "#" in method:
#         import re
#         sv_samples = int(re.findall(r"\d+", method)[0])
#
#     data, target = None, None
#     for _data, _targer in self.data_loader:
#         data = torch.cat((data, _data), 0)
#         target = torch.cat((target, _targer), 0)
#     data, target = data.to(self.device), target.to(self.device)
#
#     print (f"Data shape for SV: {data.shape}")
#
#     print("SV, samples ", sv_samples)
#     for alpha in np.linspace(0, 1, sv_samples):
#         # try:
#         #     data, target = next(data_iter)
#         # except StopIteration:
#         #     # StopIteration is thrown if dataset ends
#         #     # reinitialize data loader
#         #     data_iter = iter(self.data_loader)
#         #     data, target = next(data_iter)
#
#         # Compute activations of current module
#         activations, _, __ = self._run_forward(
#             data, return_intermediate_output_module=module, reduction="none"
#         )
#         activations.retain_grad()
#
#         current_act = alpha * activations
#
#         # Compute accuracy with no players
#         _, accuracy, loss = self._run_forward(
#             x=current_act, y_true=target, process_as_intermediate_output_module=module,reduction="none"
#         )
#         loss.mean().backward()
#
#         if sv is None:
#             sv = (activations * activations.grad.data).clone().detach().cpu().numpy()
#         else:
#             sv += (activations * activations.grad.data).clone().detach().cpu().numpy()
#
#
#     sv = sv / sv_samples
#     sv *= -1.
#     while len(sv.shape) >= 3:
#         sv = sv.sum(-1)
#     if "abs" in method:
#         sv = np.abs(sv)
#     if "p-one" in method:
#         sv = np.percentile(sv, 75, axis=0)
#     elif "p-two" in method:
#         sv = np.percentile(sv, 99, axis=0)
#     elif "p-three" in method:
#         sv = np.max(sv, axis=0)
#     else:
#         sv = sv.mean(0)
#
#     if self.verbose > 0:
#         assert not np.isnan(sv).any(), "nans!!"
#         print(f"Estimating Integrated Gradients for {n} players in {module}")
#         print(f"IG sum to {sv.sum()}")
#
#     return sv
