import torch
import numpy as np


activations = None
activations_count = None
gradients = None
gradients_x_input = None


def attributions_for_module(self, module, method):
    global activations, activations_count, gradients, gradients_x_input

    n = module.weight.shape[0]

    if method == "sv":
        return _fast_estimate_sv_for_module(self, module)
    if method == "weight":
        return _sum_absolute_weights(module)
    elif method == "random":
        return np.random.random((n,))

    activations = torch.zeros((n,)).to(self.device)
    activations_count = torch.zeros((n,)).to(self.device)
    gradients = torch.zeros((n,)).to(self.device)
    gradients_x_input = torch.zeros((n,)).to(self.device)

    h1 = module.register_forward_hook(_forward_hook)
    h2 = module.register_backward_hook(_backward_hook)

    data, target = next(iter(self.data_loader))
    data, target = data.to(self.device), target.to(self.device)

    # Compute accuracy with no players
    _, _, loss = self._run_forward(x=data, y_true=target,)
    loss.backward()

    h1.remove()
    h2.remove()

    if method == "grad":
        return gradients.detach().cpu().numpy()
    elif method == "taylor":
        return gradients_x_input.detach().cpu().numpy()
    elif method == "count":
        return activations_count.detach().cpu().numpy()


def _forward_hook(module, input, output):
    global activations, activations_count
    activations_count += (output > 0).sum(0).float()
    activations = output


def _backward_hook(module, grad_input, grad_output):
    global gradients, gradients_x_input
    gradients += grad_output[0].abs().sum(0)
    gradients_x_input += (grad_output[0] * activations).abs().sum(0)


def _sum_absolute_weights(module):
    w = module.weight.detach().cpu().numpy()
    w = np.abs(w)
    while len(w.shape) > 1:
        w = w.sum(-1)
    return w


def _fast_estimate_sv_for_module(self, module, sv_samples=5):
    # Estimate Shapley Values of module's output nodes
    n = module.weight.shape[0]

    with torch.no_grad():
        data, target = next(iter(self.data_loader))
        data, target = data.to(self.device), target.to(self.device)

        # Compute activations of current module
        activations, _, __ = self._run_forward(
            data, return_intermediate_output_module=module
        )

        # Compute accuracy with no players
        _, full_accuracy, full_loss = self._run_forward(
            x=activations, y_true=target, process_as_intermediate_output_module=module,
        )
        _, no_player_accuracy, no_player_loss = self._run_forward(
            x=torch.zeros_like(activations),
            y_true=target,
            process_as_intermediate_output_module=module,
        )

        # Start SV sampling
        sv = np.zeros((n,))
        count = np.ones((n,)) * 0.0001

        for j in range(sv_samples):
            # if self.verbose > 0:
            #     print(f"SV - Iteration {j}")
            _activation = activations.clone().detach()
            _, accuracy, loss = self._run_forward(
                x=_activation,
                y_true=target,
                process_as_intermediate_output_module=module,
            )
            for i in np.random.permutation(n):  # [:min(n, 1000)]:
                _activation = _activation.index_fill_(
                    1, torch.tensor(np.array([i])).long().to(self.device), 0.0
                )
                _, new_accuracy, new_loss = self._run_forward(
                    x=_activation,
                    y_true=target,
                    process_as_intermediate_output_module=module,
                )
                sv[i] += loss - new_loss
                count[i] += 1
                loss = new_loss
                accuracy = new_accuracy
        sv /= count
        sv *= -1

        if self.verbose > 0:
            print(f"Estimating Shapley Values for {n} players in {module}")
            print(f"Shapley Values sum to {sv.sum()}")
            print(f"This should match the loss gap {no_player_loss - full_loss}")

        return sv
