import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
import logging

from abc import ABC, abstractmethod

SUPPORTED_OUT_PRUNING_MODULES = [nn.Linear, _ConvNd]


class _AttributionMetric(ABC):
    def __init__(self, model, data_generator, criterion, device):
        self.model = model
        self.data_gen = data_generator
        self.criterion = criterion
        self.device = device

    @abstractmethod
    def run(self, modules):
        for m in modules:
            assert any([isinstance(m, t) for t in SUPPORTED_OUT_PRUNING_MODULES]),\
                f"Attributions can be computed only for the following modules {SUPPORTED_OUT_PRUNING_MODULES}"

    def run_forward(self, loss_reduction="mean"):
        assert loss_reduction == "mean" or loss_reduction == "none"
        cumulative_loss = None
        for idx, (x, y) in enumerate(self.data_gen):
            x = x.to(self.device)
            loss = self.criterion(self.model(x), y, reduction="none")
            if cumulative_loss is None:
                cumulative_loss = loss
            else:
                cumulative_loss = torch.cat((cumulative_loss, loss), 0)
        if loss_reduction == "mean":
            cumulative_loss = cumulative_loss.mean(0)
        return cumulative_loss

    def run_forward_and_backward(self):
        for idx, (x, y) in enumerate(self.data_gen):
            x = x.to(self.device)
            loss = self.criterion(self.model(x), y)
            loss.backward()

    def aggregate_over_samples(self, attributions, reduction="mean"):
        if reduction == "mean":
            attr = np.mean(attributions, 0)
            ordering = np.argsort(attr)
            return attr, ordering
