import numpy as np
from ..attributions import _AttributionMetric


class TaylorAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute first-order Taylor expansion of the loss

    Reference:
    Molchanov et al., Pruning convolutional neural networks for resource efficient inference
    """

    def __init__(self, *args, signed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.signed = signed

    def run(self, module):
        super().run(module)
        handles = [module.register_forward_hook(self._forward_hook()),
                   module.register_backward_hook(self._backward_hook())]
        self.run_all_forward_and_backward()
        attr = module._tp_taylor
        result = self.aggregate_over_samples(attr)
        delattr(module, "_tp_taylor")
        for h in handles:
            h.remove()
        return result

    @staticmethod
    def _forward_hook():
        def _hook(module, _, output):
            module._tp_activation = output
        return _hook

    def _backward_hook(self):
        def _hook(module, _, grad_output):
            taylor = -1. * (grad_output[0] * module._tp_activation)
            if len(taylor.shape) > 2:
                taylor = taylor.flatten(2).sum(-1)
            if self.signed is False:
                taylor = taylor.abs()
            if not hasattr(module, "_tp_taylor"):
                module._tp_taylor = taylor.detach().cpu().numpy()
            else:
                module._tp_taylor = np.concatenate((module._tp_taylor, taylor.detach().cpu().numpy()), 0)
        return _hook




