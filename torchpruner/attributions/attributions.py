import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import logging

from abc import ABC, abstractmethod

SUPPORTED_OUT_PRUNING_MODULES = [nn.Linear, _ConvNd]


class _AttributionMetric(ABC):
    def __init__(self, model, data_generator, criterion, device, reduction="mean"):
        assert reduction in ["mean", "none", "sum"] or callable(reduction), \
            'Reduction must be a string in ["mean", "none", "sum"] or a function'
        self.model = model
        self.data_gen = data_generator
        self.criterion = criterion
        self.device = device
        self.reduction = reduction

    @abstractmethod
    def run(self, module):
        assert any(
            [isinstance(module, t) for t in SUPPORTED_OUT_PRUNING_MODULES]
        ), f"Attributions can be computed only for the following modules {SUPPORTED_OUT_PRUNING_MODULES}"

    def run_all_forward(self):
        """
        Run forward pass on all data in `data_gen`, returning loss for each example
        :return: Tensor
        """
        cumulative_loss = None
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data_gen):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.criterion(self.model(x), y, reduction="none")
                if cumulative_loss is None:
                    cumulative_loss = loss
                else:
                    cumulative_loss = torch.cat((cumulative_loss, loss), 0)
            return cumulative_loss

    def run_all_forward_and_backward(self):
        """
        Run forward and backward passes on all data in `data_gen`
        :return: None
        """
        for idx, (x, y) in enumerate(self.data_gen):
            x, y = x.to(self.device), y.to(self.device)
            loss = self.criterion(self.model(x), y)
            loss.backward()

    def run_forward_partial(
        self, x=None, y_true=None, to_module=None, from_module=None
    ):
        """
        Run the forward pass on a given data `x`. If target is provided, also computes and
        returns loss. This function assumes the model is equipped with `forward_partial`
        method to run only part of the computational graph.
        :param x:
        :param y_true:
        :param to_module:
        :param from_module:
        :return:
        """
        loss = None
        y = self.model.forward_partial(x, to_module=to_module, from_module=from_module,)
        if y_true is not None and to_module is None:
            loss = self.criterion(y, y_true, reduction="none")
        return y, loss

    def aggregate_over_samples(self, attributions):
        """
        Aggregate the attribution computed on each input example according to some reduction.
        While most often the mean is used, there are cases where a different aggregation might
        be preferred.
        :param attributions:
        :return:
        """
        if self.reduction == "mean":
            return np.mean(attributions, 0)
        elif self.reduction == "sum":
            return np.sum(attributions, 0)
        elif self.reduction == "none":
            return attributions
        else:  # a function
            return self.reduction(attributions)
