import numpy as np
import torch
from ..attributions import _AttributionMetric


class APoZAttributionMetric(_AttributionMetric):
    """
    Compute attributions as 1-APoZ (Average Percentage of Zeros)

    Reference:
    Hu et al., Net-work trimming: A data-driven neuron pruning approach
    towards  efficient  deep  architectures
    """

    def run(self, modules):
        super().run(modules)
        with torch.no_grad():
            result = []
            handles = []
            for m in modules:
                handles.append(m.register_forward_hook(self._forward_hook()))
            self.run_forward()
            for m in modules:
                attr = m._tp_nonzero_count
                result.append(self.aggregate_over_samples(attr))
                delattr(m, "_tp_nonzero_count")
            for h in handles:
                h.remove()
            return result

    @staticmethod
    def _forward_hook():
        def _hook(module, _, output):
            nonzero_count = (output > 0).float()
            while len(nonzero_count.shape) > 2:
                nonzero_count = nonzero_count.sum(-1)
            if not hasattr(module, "_tp_nonzero_count"):
                module._tp_nonzero_count = nonzero_count.detach().cpu().numpy()
            else:
                module._tp_nonzero_count = np.concatenate(
                    (module._tp_nonzero_count, nonzero_count.detach().cpu().numpy()), 0)
        return _hook
