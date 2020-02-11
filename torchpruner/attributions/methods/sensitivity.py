import torch
from ..attributions import _AttributionMetric


class SensitivityAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute gradient of the loss

    Reference:
    Mittal et al., Studying the plasticity in deep convolutional neural networks using random pruning
    """

    def run(self, modules):
        super().run(modules)
        result = []
        for m in modules:
            m.register_backward_hook(self._backward_hook())
        self.run_forward_and_backward()
        for m in modules:
            attr = m._tp_gradient.detach().cpu().numpy()
            result.append(self.aggregate_over_samples(attr))
        return result

    @staticmethod
    def _backward_hook():
        def _hook(module, _, grad_output):
            grad = grad_output[0].abs()
            if len(grad.shape) > 2:
                grad = grad.flatten(2).sum(-1)
            if not hasattr(module, "_tp_gradient"):
                module._tp_gradient = grad
            else:
                module._tp_gradient = torch.cat((module._tp_gradient, grad), 0)
        return _hook