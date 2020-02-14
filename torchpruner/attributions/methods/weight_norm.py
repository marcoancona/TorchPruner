import numpy as np
from ..attributions import _AttributionMetric


class WeightNormAttributionMetric(_AttributionMetric):
    """
    Compute attributions as sum of absolute weight.

    Reference:
    Li et al., Pruning filters for efficient convnets, ICLR 2017
    """

    def run(self, module):
        super().run(module)
        attr = module.weight.detach().cpu().numpy()
        attr = np.abs(attr)
        while len(attr.shape) > 1:
            attr = attr.sum(-1)
        return attr