import torch
import numpy as np
from ..attributions import _AttributionMetric


class ShapleyAttributionMetric(_AttributionMetric):
    """
    Compute attributions as Shapley values
    """

    def __init__(self, model, data_generator, criterion, device, sv_samples=5):
        super().__init__(model, data_generator, criterion, device)
        self.samples = sv_samples
        self.mask_indices = []

    def run(self, modules, sv_samples=None):
        super().run(modules)
        result = []
        for m in modules:
            result.append(
                self.run_module(
                    m, sv_samples if sv_samples is not None else self.samples
                )
            )
        return result

    def run_module(self, module, samples):
        with torch.no_grad():
            n = module.weight.shape[0]  # output dimension
            original_loss = self.run_forward(loss_reduction="none")
            handle = module.register_forward_hook(self._forward_hook())

            sv = torch.zeros((original_loss.shape[0], n)).to(self.device)
            for _ in range(samples):
                self.mask_indices = []
                loss = original_loss.detach().clone()
                for i in np.random.permutation(n):
                    self.mask_indices.append(i)
                    new_loss = self.run_forward(loss_reduction="none")
                    sv[:, i] += ((new_loss - loss) / samples).squeeze()
                    loss = new_loss

            # zero_player_loss = self.run_forward(loss_reduction="none")
            # print ((zero_player_loss-original_loss).mean(0))
            # print (sv.mean(0).sum())
            handle.remove()
            return self.aggregate_over_samples(sv.detach().cpu().numpy())

    def _forward_hook(self):
        def _hook(module, _, output):
            return output.index_fill_(
                1, torch.tensor(self.mask_indices).to(self.device), 0.0,
            )

        return _hook
