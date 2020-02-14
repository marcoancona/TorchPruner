import numpy as np
from ..attributions import _AttributionMetric


class SensitivityAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute gradient of the loss

    Reference:
    Mittal et al., Studying the plasticity in deep convolutional neural networks using random pruning
    """

    def run(self, module):
        super().run(module)
        handles = [module.register_backward_hook(self._backward_hook())]
        self.run_all_forward_and_backward()
        attr = module._tp_gradient
        result = self.aggregate_over_samples(attr)
        delattr(module, "_tp_gradient")
        for h in handles:
            h.remove()
        return result

    @staticmethod
    def _backward_hook():
        def _hook(module, _, grad_output):
            grad = grad_output[0].abs()
            if len(grad.shape) > 2:
                grad = grad.flatten(2).sum(-1)
            if not hasattr(module, "_tp_gradient"):
                module._tp_gradient = grad.detach().cpu().numpy()
            else:
                module._tp_gradient = np.concatenate((module._tp_gradient, grad.detach().cpu().numpy()), 0)
        return _hook