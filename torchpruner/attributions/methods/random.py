import numpy as np
from ..attributions import _AttributionMetric


class RandomAttributionMetric(_AttributionMetric):
    def run(self, module):
        super().run(module)
        n = module.weight.shape[0]  # output dimension
        return np.random.random((n,))

