import numpy as np
from ..attributions import _AttributionMetric


class RandomAttributionMetric(_AttributionMetric):
    def run(self, module, **kwargs):
        module = super().run(module, **kwargs)
        n = module.weight.shape[0]  # output dimension
        return np.random.random((n,))

    def find_evaluation_module(self, module, find_best_evaluation_module=False):
        # Needs to work on module with weights
        return module

