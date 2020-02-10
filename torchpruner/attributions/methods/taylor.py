import torch
from ..attributions import _AttributionMetric


class TaylorAttributionMetric(_AttributionMetric):
    """
    Compute attributions as average absolute first-order Taylor expansion of the loss

    Reference:
    Molchanov et al., Pruning convolutional neural networks for resource efficient inference
    """

    def __init__(self, signed_attribution=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signed = signed_attribution

    def run(self, modules):
        super(TaylorAttributionMetric).run(modules)
        result = []
        for m in modules:
            m.register_forward_hook(self._forward_hook())
            m.register_backward_hook(self._backward_hook())
        self.run_forward()
        for m in modules:
            attr = m._tp_taylor.deatch().cpu().numpy()
            result.append(self.aggregate_over_samples(attr))
        return result

    @staticmethod
    def _forward_hook():
        def _hook(module, _, output):
            module._tp_activation = output
        return _hook

    def _backward_hook(self):
        def _hook(module, _, grad_output):
            taylor = (grad_output[0] * module._tp_activation)
            if len(taylor.shape) > 2:
                grad_x_input = taylor.flatten(2).sum(-1)
            if self.signed is False:
                taylor = taylor.abs()
            if module._tp_taylor is None:
                module._tp_taylor = taylor
            else:
                module._tp_taylor = torch.cat((module._tp_taylor, taylor), 0)
        return _hook




