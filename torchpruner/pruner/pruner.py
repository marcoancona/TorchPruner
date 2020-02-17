import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.dropout import _DropoutNd
from torchpruner.pruner.opt_pruner import OptimizerPruner
import logging

SUPPORTED_IN_PRUNING_MODULES = [nn.Linear, _ConvNd, nn.Dropout, _BatchNorm]
SUPPORTED_OUT_PRUNING_MODULES = [nn.Linear, _ConvNd]


class Pruner:
    def __init__(self, model, input_size, device, optimizer=None):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.optimizer = optimizer

    def prune_model(self, module, indices, cascading_modules=None):
        """
        :param pruning_graph:
        :return:
        """
        # Implement nan trick
        # 1. Install forward hook to simulate pruning by settings activations to nan
        module_handle = module.register_forward_hook(self._nanify_hook(indices))

        # 2. Get all potentially prunable modules, if not provided by the user
        if cascading_modules is None:
            print ("Warning: no cascading modules defined")
            cascading_modules = []

        # 3. Install a nan listener on all modules
        handles = []
        for next_module in cascading_modules:
            handles.append(next_module.register_forward_hook(self._detect_nan_hook()))

        # 4. Run a forward pass recording nans
        self._run_forward()
        module_handle.remove()
        for handle in handles:
            handle.remove()

        # 5. Prune all cascading modules where nans have been detected
        for next_module in cascading_modules:
            if hasattr(next_module, "_nan_indices"):
                self.prune_module(
                    next_module, getattr(next_module, "_nan_indices"),
                    direction="in",
                    original_len=getattr(next_module, "_activation_len")
                )
                delattr(next_module, "_nan_indices")

        # 5. Finally, prune module
        self.prune_module(module, indices, direction="out")

    def prune_module(self, module, indices, direction="out", original_len=None):
        """
        Prune a module parameters. This method provides an higher level API for
        prune_parameter with understanding of the module class and its corresponding
        parameters name.
        :param module:
        :param indices:
        :param direction:
        :return:
        """
        assert direction in ["out", "in"], "direction should be 'out' or 'in'"
        if direction is "out":
            assert any(
                [isinstance(module, t) for t in SUPPORTED_OUT_PRUNING_MODULES]
            ), f"Cannot prune outgoing activations on this module. Only the following are supported {SUPPORTED_OUT_PRUNING_MODULES}"
        else:
            assert any(
                [isinstance(module, t) for t in SUPPORTED_IN_PRUNING_MODULES]
            ), f"Cannot prune incoming activations on this module. Only the following are supported {SUPPORTED_IN_PRUNING_MODULES}"

        print(f"Pruning {len(indices)} units from {module} ({direction})")
        if direction is "out":
            self.prune_parameter(module, "weight", indices, axis=0)
            self.prune_parameter(module, "bias", indices, axis=0)
        else:
            if isinstance(module, nn.Linear) or isinstance(module, _ConvNd):
                self.prune_parameter(module, "weight", indices, axis=1)
            elif isinstance(module, _BatchNorm):
                self.prune_parameter(module, "weight", indices, axis=0)
                self.prune_parameter(module, "bias", indices, axis=0)
                self.prune_parameter(module, "running_mean", indices, axis=0)
                self.prune_parameter(module, "running_var", indices, axis=0)
            elif isinstance(module, _DropoutNd):
                self._adjust_dropout(module, indices, original_len)

    def prune_parameter(self, module, parameter_name, indices, axis=0):
        """
        Prune a single parameter Tensor within a module
        :param module:
        :param parameter_name:
        :param indices:
        :param axis:
        :return:
        """
        param = getattr(module, parameter_name)
        if param is not None:
            n = param.data.shape[axis]
            mask = np.ones(n, dtype=bool)
            mask[indices] = False
            keep_indices = torch.tensor(np.arange(n)[mask]).to(self.device)
            param.data = param.data.index_select(axis, keep_indices)
            # If gradient is not None, we need to slice it as well
            if param.grad is not None:
                param.grad.data = param.grad.data.index_select(axis, keep_indices)
            # Finally, might need to slice other parameters in the optimizer
            if self.optimizer is not None:
                OptimizerPruner.prune(self.optimizer, param, axis, keep_indices, self.device)

    def _adjust_dropout(self, module, indices, original_len):
        """
        Adjust dropout ratio, such that the average number of active units will
        be the same after pruning
        :param module:
        :param pruning_ratio:
        :return:
        """
        if original_len is None:
            raise RuntimeError("Cannot adjust Dropout rate with 'original_len=None'")
        module.p *= (1.0 - len(indices) / original_len)

    def _nanify_hook(self, indices):
        """
        Hook to set nans long dim 1 of output Tensor to a module
        (simulated pruning)
        :param indices:
        :return:
        """

        def _hook(_, __, output):
            return output.index_fill_(
                1,
                torch.tensor(indices).to(self.device),
                torch.tensor(np.nan).to(self.device),
            )

        return _hook

    @staticmethod
    def _detect_nan_hook():
        """
        Hook to detect nans along dim 1 of input Tensor to a module
        :return:
        """

        def _hook(module, input, __):
            input = input[0]
            setattr(
                module, "_activation_len", float(input.shape[1])
            )
            while len(input.shape) > 2:
                input = input.sum(-1)
            input = input.sum(0).flatten(0)
            indices = (
                torch.isnan(input).nonzero().flatten(0).detach().clone().cpu().numpy()
            )
            if len(indices) > 0:
                setattr(
                    module, "_nan_indices", indices
                )
        return _hook

    def _run_forward(
        self, x=None,
    ):
        d, b = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if x is None:
            x = (
                torch.tensor(np.random.random((2,) + self.input_size))
                .float()
                .to(self.device)
            )
        y = self.model(x)
        torch.backends.cudnn.deterministic = d
        torch.backends.cudnn.benchmark = b
        return y
