import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import json
from torchsummary import summary
import matplotlib.pyplot as plt
from experiments.utils import log_dict
from shapley_pruning.attributions import attributions_for_module


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
        experiment_id="experiment",
    ):
        self.model = model
        # self.prevent_pruning = prevent_pruning if prevent_pruning is not None else []
        # self.pruning_deps = pruning_dependencies if pruning_dependencies is not None else {}
        self.performing_pruning = False
        self.input_size = input_size
        self.pruning_graph = pruning_graph if pruning_graph is not None else {}
        self.prunable_layers = []
        self.activations = {}
        self.activation_cum_grad = {}
        self.activation_counts = {}
        self.activation_cum_taylor = {}
        self.device = device if isinstance(type, torch.device) else torch.device(device)
        # self.outputs = {}
        self.pruning_chain = {}
        self.verbose = verbose
        self.data_loader = data_loader
        self.test_data_load = test_data_loader
        self.loss = loss
        self.experiment_id = experiment_id

        self._module_name_cache = {}
        self._ranking_method = None
        self._max_loss_increase = None
        self._epoch = 0

        self._run_forward()

    def prune(
        self,
        sparsity_ratio,
        ranking_method,
        optimizer,
        max_loss_increase_percent=None,
        epoch=0,
        prune_all_layers=False,
    ):
        """
        Remove a given percentage of nodes from the network.
        If global_ranking=True, all prunable nodes will be ranked together, otherwise
        percentage nodes will be removed from each prunable module independently.
        :param percentage:
        :param global_ranking:
        :param ranking_method:
        :return:
        """

        self._ranking_method = ranking_method
        self._max_loss_increase = max_loss_increase_percent
        self._epoch = epoch

        remove_ratio = 1.0 - sparsity_ratio if sparsity_ratio > 0 else None
        assert remove_ratio is not None or max_loss_increase_percent is not None
        assert not (remove_ratio is not None and max_loss_increase_percent is not None)
        self.performing_pruning = True
        self.opt_state_dict = optimizer.state_dict()

        # Prune layers one at the time
        if prune_all_layers:
            to_prune = self.pruning_graph
        else:
            current_module_idx = max(0, (epoch - 1)) % len(self.pruning_graph)
            to_prune = self.pruning_graph[current_module_idx : current_module_idx + 1]

        for module, cascading_modules in to_prune:

            if self.verbose > 0:
                print(f"Pruning {module}")

            # Estimates scores for the activations of the current module
            indices_to_remove = self._get_pruning_indices(
                module,
                ranking_method,
                pruning_ratio=remove_ratio,
                max_loss_increase_percent=max_loss_increase_percent,
            )

            if self.verbose > 0:
                print(f"Removing {len(indices_to_remove)} activations")

            # First, prune cascading modules
            # Note: we need to prune first cascading modules because we need to be able to first run
            # a forward pass to compute the pruning mask of the cascading modules
            for next_module in cascading_modules:
                if self.verbose > 0:
                    print(f"-->\tCascade pruning {next_module}")
                indices_list = self._get_indices_mapping_for_pruning(
                    module, next_module, indices_to_remove
                )
                self._prune_module(next_module, indices_list)

            # Then proceed with pruning of the current module
            self._prune_module(
                module,
                [("weight", 0, indices_to_remove), ("bias", 0, indices_to_remove)],
            )

            if self.verbose > 0:
                summary(self.model, input_size=self.input_size, device=self.device.type)

        # Reset all statistics
        self._zero_gradients()
        self.activation_cum_grad = {}
        self.activation_cum_taylor = {}
        self.activation_counts = {}
        self.activations = {}
        # Just to test that the new network works, we run a forward pass. TODO: remove
        self._run_forward()
        self._zero_gradients()
        self.performing_pruning = False

        # Important! 'params' in opt state must be sorted as the model parameters to load
        # everything correctly. Took me ages to find out the problem.
        model_parameters_ids = [id(p) for p in self.model.parameters()]
        self.opt_state_dict["param_groups"][0]["params"] = model_parameters_ids
        return self.opt_state_dict

    def _module_name(self, module):
        if module in self._module_name_cache:
            return self._module_name_cache[module]

        for module_name, m in self.model.named_modules():
            if module == m:
                self._module_name_cache[module] = module_name
                return module_name
        return None

    def _get_pruning_indices(
        self, module, ranking_method, pruning_ratio, max_loss_increase_percent
    ):
        scores, indices = None, None

        if ranking_method.startswith("random"):
            scores = attributions_for_module(self, module, "random")
        elif ranking_method.startswith("grad"):
            scores = attributions_for_module(self, module, "grad")
        elif ranking_method.startswith("weight"):
            scores = attributions_for_module(self, module, "weight")
        elif ranking_method.startswith("taylor"):
            scores = attributions_for_module(self, module, "taylor")
        elif ranking_method.startswith("count"):
            scores = attributions_for_module(self, module, "count")
        elif ranking_method.startswith("sv"):
            scores = attributions_for_module(self, module, "sv")

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
            # Dynamic pruning (how many to remove is based on max_loss_increase_percent)
            indices, loss_increment = self._compute_dynamic_pruning_indices(
                module, scores, max_loss_increase_percent
            )

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

    def _get_indices_mapping_for_pruning_fixed(
        self, module, next_module, pruning_indices
    ):
        if len(pruning_indices) == 0:
            return np.array([])

        assert any(
            [isinstance(module, t) for t in [nn.Linear, nn.Conv2d]]
        ), "Only Linear and Conv2D supported for pruning"

        assert any(
            [
                isinstance(module, t)
                for t in [nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d]
            ]
        ), "Only Linear and Conv2D supported for cascading pruning"

        def _get_masked_output(module, pruning_indices):
            original_weights = module.weight.clone().detach()
            module.weight.requires_grad = False
            module.weight.index_fill_(
                0,
                torch.tensor(pruning_indices).to(self.device),
                torch.tensor(np.nan).to(self.device),
            )
            module.weight.requires_grad = True
            activations, _, __ = self._run_forward(
                return_intermediate_output_module=module
            )
            module.weight.data = original_weights
            module.weight.requires_grad = True
            return activations

        default = [("weight", 1, pruning_indices)]

        if isinstance(module, nn.Conv2d):
            conv_activations = _get_masked_output(module, pruning_indices)
            lin_activations = torch.flatten(conv_activations, 1).sum(0)
            mask_indices = np.argwhere(
                np.isnan(lin_activations.clone().detach().cpu().numpy())
            ).flatten()
            if isinstance(next_module, nn.Linear):
                return [("weight", 1, mask_indices)]
            elif isinstance(next_module, nn.BatchNorm2d):
                return [
                    ("weight", 0, mask_indices),
                    ("bias", 0, mask_indices),
                    ("running_mean", 0, mask_indices),
                    ("running_std", 0, mask_indices),
                ]

        elif isinstance(module, nn.Linear):
            if isinstance(next_module, nn.BatchNorm1d):
                return [
                    ("weight", 0, pruning_indices),
                    ("bias", 0, pruning_indices),
                    ("running_mean", 0, pruning_indices),
                    ("running_std", 0, pruning_indices),
                ]

        return default

    def _get_indices_mapping_for_pruning(self, module, next_module, pruning_indices):
        if len(pruning_indices) == 0:
            return np.array([])
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
        self._zero_gradients()
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

        print(f"Found {torch.isnan(activations).sum()} nan activations")
        # print (f"-- {activations.shape} act shape")
        activations.sum().backward()
        # print(next_module.weight.grad)
        # print (next_module.weight.grad.requires_grad)
        grad = next_module.weight.grad.clone().detach().cpu().numpy()
        # print(f"Found {torch.isnan(next_module.weight.grad).sum()} nan gradients")
        if len(grad.shape) > 1:
            grad = grad.sum(0)
            while len(grad.shape) > 1:
                grad = grad.sum(-1)
        mask_indices = np.argwhere(np.ma.mask_or(np.isnan(grad), np.isinf(grad)))
        if len(mask_indices) > 0:
            if len(mask_indices[0].shape) == 1:
                indices = np.unique(mask_indices.flatten())
            else:
                indices = np.unique(mask_indices[:, 1].flatten())
        else:
            indices = np.array([])

        module.weight.data = original_weights
        module.weight.requires_grad = True
        if isinstance(next_module, nn.Linear) or isinstance(next_module, nn.Conv2d):
            return [("weight", 1, indices)]
        else:
            # BatchNorm2D/1D
            return [
                ("weight", 0, indices),
                ("bias", 0, indices),
                ("running_mean", 0, indices),
                ("running_var", 0, indices),
            ]

    def _compute_dynamic_pruning_indices(self, module, scores, max_increase_percent):
        with torch.no_grad():
            # Compute activations of current module

            data, target = next(iter(self.data_loader))
            data, target = data.to(self.device), target.to(self.device)
            activations, _, __ = self._run_forward(
                data, return_intermediate_output_module=module
            )
            _, full_accuracy, full_loss = self._run_forward(
                x=activations,
                y_true=target,
                process_as_intermediate_output_module=module,
            )
            all_indices = np.argsort(scores)
            indices = []
            loss_history = []
            score_history = []
            new_loss = full_loss
            k = None
            for i in all_indices:

                _, new_accuracy, new_loss = self._run_forward(
                    x=activations.index_fill(
                        1,
                        torch.tensor(np.array(indices + [i])).long().to(self.device),
                        0.0,
                    ),
                    y_true=target,
                    process_as_intermediate_output_module=module,
                )
                indices.append(i)
                loss_history.append(new_loss.detach().cpu().numpy().item())
                score_history.append(scores[i])

                if 100 * (new_loss - full_loss) / full_loss <= max_increase_percent:
                    continue
                else:
                    # Normally would break here, but we want the full statistic to plot it
                    if k is None:
                        k = len(indices)

            # Statistics
            log_dict(
                f"{self.experiment_id}",
                {
                    "ranking_method": self._ranking_method,
                    "layer": self._module_name(module),
                    "max_loss_increase": self._max_loss_increase,
                    "epoch": self._epoch,
                    "full_loss": full_loss.detach().cpu().numpy().item(),
                    "loss_history": json.dumps(loss_history),
                    "indices": json.dumps(np.array(indices).tolist()),
                    "scores": json.dumps(np.array(score_history).tolist()),
                    "k": k,
                },
            )

            return indices[:k], new_loss - full_loss

    def _prune_module(self, module, indices_list):
        for param_name, axis, indices in indices_list:
            if hasattr(module, param_name):
                param = module.__getattr__(param_name)
                if param is not None:
                    L = param.shape[axis]
                    mask = np.ones(L, dtype=bool)  # all elements included/True.
                    mask[indices] = False
                    keep_indices = np.arange(L)[mask]

                    if self.verbose > 0:
                        print(
                            f"Pruning {len(indices)} ({len(indices) / L * 100:.1f}%) nodes on '{param_name}' of {module}'"
                        )

                    old_id = id(param)
                    with torch.no_grad():
                        setattr(
                            module,
                            param_name,
                            nn.Parameter(
                                param.index_select(
                                    axis, torch.tensor(keep_indices).to(self.device)
                                ),
                                requires_grad=param.requires_grad,
                            ),
                        )
                        param = module.__getattr__(param_name)
                        new_id = id(param)
                        self._update_optimizer(
                            old_id, new_id, axis, keep_indices, param
                        )

    def _update_optimizer(self, old_id, new_id, prune_axis, keep_indices, new_weight):
        if old_id in self.opt_state_dict["state"]:
            momentum = self.opt_state_dict["state"][old_id]["momentum_buffer"]
            momentum = momentum.index_select(
                prune_axis, torch.tensor(keep_indices).to(self.device)
            )
            assert momentum.shape == new_weight.shape
            assert new_id == id(new_weight)
            del self.opt_state_dict["state"][old_id]
            self.opt_state_dict["state"][new_id] = {
                "momentum_buffer": momentum,
            }
            self.opt_state_dict["param_groups"][0]["params"].remove(old_id)
            self.opt_state_dict["param_groups"][0]["params"].append(new_id)

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
