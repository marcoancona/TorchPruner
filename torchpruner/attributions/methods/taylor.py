import torch
from ..attributions import _AttributionMetric


class TaylorAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute first-order Taylor expansion of the loss

    Reference:
    Molchanov et al., Pruning convolutional neural networks for resource efficient inference
    """

    def __init__(self, model, data_generator, criterion, device, signed_attribution=False):
        super().__init__(model, data_generator, criterion, device)
        self.signed = signed_attribution

    def run(self, modules):
        super().run(modules)
        result = []
        for m in modules:
            m.register_forward_hook(self._forward_hook())
            m.register_backward_hook(self._backward_hook())
        self.run_forward_and_backward()
        for m in modules:
            attr = m._tp_taylor.detach().cpu().numpy()
            result.append(self.aggregate_over_samples(attr))
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
                module._tp_taylor = taylor
            else:
                module._tp_taylor = torch.cat((module._tp_taylor, taylor), 0)
        return _hook




