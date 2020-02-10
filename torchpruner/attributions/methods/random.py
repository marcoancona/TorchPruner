import numpy as np
from ..attributions import _AttributionMetric


class RandomAttributionMetric(_AttributionMetric):
    def run(self, modules):
        super().run(modules)
        result = []
        for m in modules:
            n = m.weight.shape[0]  # output dimension
            attr = np.random.random((n,))
            result.append((attr, np.argsort(attr)))
        return result
