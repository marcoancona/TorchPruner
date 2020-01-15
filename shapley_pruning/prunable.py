import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class ContinuousPruner:
    def __init__(
        self,
        model,
        input_size,
        pruning_graph,
        device,
        data_loader,
        test_data_loader,
        loss,
        verbose=0,
    ):
        self.model = model
        # self.prevent_pruning = prevent_pruning if prevent_pruning is not None else []
        # self.pruning_deps = pruning_dependencies if pruning_dependencies is not None else {}
        self.performing_pruning = False
        self.input_size = input_size
        self.pruning_graph = pruning_graph if pruning_graph is not None else {}
        self.prunable_layers = []
        self.activation_cum_grad = {}
        self.activation_counts = {}
        self.device = device
        # self.outputs = {}
        self.pruning_chain = {}
        self.verbose = verbose
        self.data_loader = data_loader
        self.test_data_load = test_data_loader
        self.loss = loss

        self._run_forward()
        self._register_hooks()

    def prune(self, sparsity_ratio, ranking_method, optimizer):
        """
        Remove a given percentage of nodes from the network.
        If global_ranking=True, all prunable nodes will be ranked together, otherwise
        percentage nodes will be removed from each prunable module independently.
        :param percentage:
        :param global_ranking:
        :param ranking_method:
        :return:
        """
        remove_ratio = 1.0 - sparsity_ratio if sparsity_ratio > 0 else None
        self.performing_pruning = True
        self.opt_state_dict = optimizer.state_dict()
        # remove_indices = self._get_pruning_indices(remove_ratio, global_ranking, ranking_method)
        for module, cascading_modules in self.pruning_graph:

            # Estimates scores for the activations of the current module
            indices_to_remove = self._get_pruning_indices(
                module, ranking_method, pruning_ratio=remove_ratio
            )

            if self.verbose > 0:
                print(f"Pruning {module}")
                print(f"Removing {len(indices_to_remove)} activations")

            # First, prune cascading modules
            # Note: we need to prune first cascading modules because we need to be able to first run
            # a forward pass to compute the pruning mask of the cascading modules
            for next_module in cascading_modules:
                indices = self._get_indices_mapping_for_pruning(
                    module, next_module, indices_to_remove
                )
                # print (indices)
                self._prune_module(next_module, "in", indices)

            # Then proceed with pruning of the current module
            self._prune_module(module, "out", indices_to_remove)

        if self.verbose > 0:
            summary(self.model, input_size=self.input_size, device=self.device.type)

        # Reset all statistics
        self._zero_gradients()
        self.performing_pruning = False
        self.activation_cum_grad = {}
        self.activation_counts = {}
        # Just to test that the new network works, we run a forward pass. TODO: remove
        self._run_forward()
        self._zero_gradients()

        # Important! 'params' in opt state must be sorted as the model parameters to load
        # everything correctly. Took me ages to find out the problem.
        model_parameters_ids = [id(p) for p in self.model.parameters()]
        self.opt_state_dict['param_groups'][0]['params'] = model_parameters_ids
        return self.opt_state_dict

    def _get_pruning_indices(self, module, ranking_method, pruning_ratio):
        scores, indices = None, None

        if ranking_method.startswith("random"):
            scores = np.random.random(module.weight.shape[0])
        elif ranking_method.startswith("grad"):
            scores = self.activation_cum_grad[module]
        elif ranking_method.startswith("count"):
            scores = self.activation_counts[module]
        elif ranking_method.startswith("sv"):
            scores = self._fast_estimate_sv_for_module(module)

        if scores is None:
            raise RuntimeError("ranking_method not valid")

        while len(scores.shape) > 1:
            print("WARNING: found scores shape with more than 1 dimension")
            scores = scores.sum(-1)

        if "-abs" in ranking_method:
            scores = np.abs(scores)

        if pruning_ratio is not None:
            # Fixed pruning ratio
            N = len(scores)
            k = int(pruning_ratio * N)
            indices = np.argsort(scores)[:k]
        else:
            # Dynamic pruning (how many to remove is not fixed)
            if "zeros" in ranking_method:
                indices = np.argwhere(np.abs(scores) < 0.001).flatten()
            elif "nonpositive" in ranking_method:
                indices = np.argwhere(scores <= 0.0).flatten()
            else:
                raise RuntimeError("Criteria for dynamic pruning not understood")

        if self.verbose > 0:
            print(f"Sum of scores of removed indices: {scores[indices].sum()}")
        return indices

    def _run_forward(
        self,
        x=None,
        y_true=None,
        return_intermediate_output_module=None,
        process_as_intermediate_output_module=None,
        linearize=False,
    ):
        acc, loss = None, None
        if x is None:
            x = (
                torch.tensor(10 * np.random.random((50,) + self.input_size))
                .float()
                .to(self.device)
            )
        # print(x.shape)
        y = self.model(
            x,
            return_intermediate_output_module=return_intermediate_output_module,
            process_as_intermediate_output_module=process_as_intermediate_output_module,
            linearize=linearize,
        )
        if y_true is not None and return_intermediate_output_module is None:
            y_pred = y.argmax(dim=1, keepdim=True)
            loss = self.loss(y, y_true, reduction="mean")
            acc = y_pred.eq(y_true.view_as(y_pred)).sum().item() / y_true.shape[0]
        return y, acc, loss

    def _get_indices_mapping_for_pruning(self, module, next_module, pruning_indices):
        if len(pruning_indices) == 0:
            return np.array([])

        self._zero_gradients()

        assert any(
            [isinstance(module, t) for t in [nn.Linear, nn.Conv2d]]
        ), "Only Linear and Conv2D supported for pruning"

        assert any(
            [
                isinstance(module, t)
                for t in [nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d]
            ]
        ), "Only Linear and Conv2D supported for cascading pruning"

        # Copy module's weights and replace the axis that will be pruned with nans
        original_weights = module.weight.clone().detach()
        module.weight.requires_grad = False
        module.weight.index_fill_(
            0,
            torch.tensor(pruning_indices).to(self.device),
            torch.tensor(np.nan).to(self.device),
        )
        module.weight.requires_grad = True

        # Produce an output up to next_module.
        # Notice that the activations will contain nans if next_module's weight interact with
        # the output of module affected by pruning

        activations, _, __ = self._run_forward(
            return_intermediate_output_module=next_module, linearize=True
        )
        # activations, _, __ = self._run_forward(
        #     x=activations_current[1:2],
        #     process_as_intermediate_output_module=module,
        #     return_intermediate_output_module=next_module,
        #     linearize=True
        # )
        # print (len(pruning_indices))
        # print (torch.isnan(activations_current[0]).sum())
        # print (torch.isnan(activations[0]).sum())

        # Find which weight indices we need to prune
        # print (activations)
        activations.sum().backward()
        grad = next_module.weight.grad.clone().detach().cpu().numpy()
        if len(grad.shape) > 1:
            # print ("Grad shape", grad.shape)
            grad = grad.sum(0)
            while len(grad.shape) > 1:
                grad = grad.sum(-1)
        mask_indices = np.argwhere(np.isnan(grad))
        # print (mask_indices.shape)
        # print(mask_indices)
        # print (len(mask_indices))
        if len(mask_indices) > 0:
            if len(mask_indices[0].shape) == 1:
                indices = np.unique(mask_indices.flatten())
            else:
                indices = np.unique(mask_indices[:, 1].flatten())
        else:
            indices = np.array([])

        module.weight.data = original_weights
        module.weight.requires_grad = True
        return indices

    def _test_loss_splot(self, module, rankings_with_name):
        loaders = [(self.data_loader, "train"), (self.test_data_load, "test")]
        with torch.no_grad():
            for loader, name in loaders:
                data, target = next(iter(loader))
                data, target = data.to(self.device), target.to(self.device)

                # Compute activations of current module
                activations, _, __ = self._run_forward(
                    data, return_intermediate_output_module=module
                )

                # Compute accuracy with no players
                _, full_accuracy, full_loss = self._run_forward(
                    x=activations,
                    y_true=target,
                    process_as_intermediate_output_module=module,
                )

                plt.figure()
                for rank, rank_name in rankings_with_name:
                    loss = [full_loss]
                    _activation = activations.clone().detach()
                    for i in rank:
                        _activation = _activation.index_fill_(
                            1, torch.tensor(np.array([i])).long().to(self.device), 0.0
                        )
                        loss.append(
                            self._run_forward(
                                x=_activation,
                                y_true=target,
                                process_as_intermediate_output_module=module,
                            )[2]
                        )

                    plt.plot(range(len(loss)), np.array(loss), label=rank_name)
                plt.legend()
                plt.savefig(f"{str(module)}_loss_{name}.png")

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
                x=activations,
                y_true=target,
                process_as_intermediate_output_module=module,
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
                if self.verbose > 0:
                    print(f"SV - Iteration {j}")
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

    def _prune_module(self, module, prune_direction, indices):
        assert (
            prune_direction is "in" or prune_direction is "out"
        ), "prune_direction must be in or out"
        if len(indices) == 0:
            return  # nothing to prune

        # Determine which axis needs to be pruned on 'weight' and on 'bias'
        prune_axis_weight, prune_axis_bias = None, None
        if prune_direction is "out":
            # First dimension of weight and bias is the activations (output) dimension
            # This is true for both linear and convolutions
            prune_axis_weight = 0
            prune_axis_bias = 0
        elif prune_direction is "in":
            # If we are pruning incoming edges, bias should not be affected
            prune_axis_weight = 1 if len(module.weight.shape) >= 2 else 0
            prune_axis_bias = None if module.weight.shape != module.bias.shape else 0

        # Compute pruning mask
        original_length = module.weight.shape[prune_axis_weight]
        mask = np.ones(original_length, dtype=bool)  # all elements included/True.
        mask[indices] = False
        keep_indices = np.arange(original_length)[mask]

        if self.verbose > 0:
            print(
                f"Pruning {len(indices)} ({len(indices) / original_length * 100:.1f}%) nodes on {module}, direction '{prune_direction}'"
            )

        # Prune replacing weights
        if prune_axis_weight is not None:
            old_id = id(module.weight)
            with torch.no_grad():
                module.weight = nn.Parameter(
                    module.weight.index_select(
                        prune_axis_weight, torch.tensor(keep_indices).to(self.device)
                    ),
                    requires_grad=True,
                )
                new_id = id(module.weight)
                self._update_optimizer(old_id, new_id, prune_axis_weight, keep_indices, module.weight)

        if prune_axis_bias is not None:
            old_id = id(module.bias)
            with torch.no_grad():
                module.bias = nn.Parameter(
                    module.bias.index_select(
                        prune_axis_bias, torch.tensor(keep_indices).to(self.device)
                    ),
                    requires_grad=True,
                )
                new_id = id(module.bias)
                self._update_optimizer(old_id, new_id, prune_axis_bias, keep_indices, module.bias)

    def _update_optimizer(self, old_id, new_id, prune_axis, keep_indices, new_weight):
        momentum = self.opt_state_dict["state"][old_id]["momentum_buffer"]
        momentum = momentum.index_select(
            prune_axis, torch.tensor(keep_indices).to(self.device)
        )
        assert momentum.shape == new_weight.shape
        assert new_id == id(new_weight)
        del self.opt_state_dict["state"][old_id]
        self.opt_state_dict["state"][new_id] = {
                'momentum_buffer': momentum,
        }
        self.opt_state_dict['param_groups'][0]['params'].remove(old_id)
        self.opt_state_dict['param_groups'][0]['params'].append(new_id)
        for p in self.opt_state_dict["state"]:
            print ( self.opt_state_dict["state"][p]["momentum_buffer"].shape)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and module in self.pruning_graph:
                # self.prunable_layers.append(module)
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)
        # print ("ContinuousPruner found the following prunable layers: ", self.prunable_layers)

    def _forward_hook(self, module, input, output):
        if not self.performing_pruning:
            output_np = output.cpu().detach().clone().numpy()
            if module not in self.activation_cum_grad:
                self.activation_counts[module] = np.zeros(output[0].shape, "int")
            self.activation_counts[module] += (output_np > 0).sum(0)

    def _backward_hook(self, module, grad_input, grad_output):
        if not self.performing_pruning:
            grad_output = grad_output[0]
            if module not in self.activation_cum_grad:
                self.activation_cum_grad[module] = np.zeros(
                    grad_output[0].shape, "float"
                )

            grad = grad_output.cpu().detach().clone().numpy()
            self.activation_cum_grad[module] += grad.sum(0)

    def _zero_gradients(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        for param in model_parameters:
            if param.grad is not None:
                param.grad.data.zero_()


def test():
    from experiments.models import fmnist

    model, _ = fmnist.get_model_with_name()

    pruner = ContinuousPruner(
        model,
        (1, 28, 28),
        {
            # model.conv1: [model.conv2],
            # model.conv2: [model.fc1],
            # model.fc1: [model.fc2],
        },
        None,
        None,
        None,
        None,
    )

    # conv1 = Prunable(nn.Conv2d(1,20,3,1))
    x = torch.tensor(10 * np.random.random((1, 1, 6, 6)).astype("float")).float()

    model.conv2.weight.data = model.conv2.weight.index_fill(
        0, torch.tensor(np.array([4, 5, 6])), torch.tensor(np.nan)
    )
    pruning_fc1 = pruner._get_indices_mapping_for_pruning(
        model.conv2, model.norm2, np.array([4, 5, 6])
    )


if __name__ == "__main__":
    test()
