import torch
import numpy as np
import logging
from ..attributions import _AttributionMetric


class ShapleyAttributionMetric(_AttributionMetric):
    """
    Compute attributions as Shapley values
    """

    def __init__(self, *args, sv_samples=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = sv_samples
        self.mask_indices = []

    def run(self, modules, sv_samples=None):
        super().run(modules)
        result = []
        for m in modules:
            # print(f"Computing Shapley values on {m}...")
            sv_samples = sv_samples if sv_samples is not None else self.samples
            if hasattr(self.model, "forward_partial"):
                # print (f"--> can run with partials")
                r = self.run_module_with_partial(m, sv_samples)
            else:
                r = self.run_module(m, sv_samples)
            result.append(r)
        return result

    def run_module_with_partial(self, module, samples):
        n = module.weight.shape[0]  # output dimension
        d = len(self.data_gen.dataset)
        sv = np.zeros((d, n))
        permutations = [np.random.permutation(n) for _ in range(samples)]
        c = 0

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data_gen):
                x, y = x.to(self.device), y.to(self.device)
                original_z, _ = self.run_forward_partial(x, to_module=module)
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)

                for j in range(samples):
                    # print (f"Sample {j}")
                    loss = original_loss.detach().clone()
                    z = original_z.clone().detach()

                    for i in permutations[j]:
                        z.index_fill_(1, torch.tensor(np.array([i])).long().to(self.device), 0.0)
                        _, new_loss = self.run_forward_partial(z, y_true=y, from_module=module)
                        delta = new_loss - loss
                        n = delta.shape[0]
                        sv[c:c+n, i] += (delta / samples).squeeze().detach().cpu().numpy()
                        loss = new_loss
                c += n

            # zero_player_loss = self.run_forward(loss_reduction="none")
            # print ((zero_player_loss-original_loss).mean(0))
            # print (sv.mean(0).sum())
            return self.aggregate_over_samples(sv)

    def run_module(self, module, samples):
        with torch.no_grad():
            n = module.weight.shape[0]  # output dimension
            original_loss = self.run_all_forward()
            handle = module.register_forward_hook(self._forward_hook())

            sv = np.zeros((original_loss.shape[0], n))
            for j in range(samples):
                # print (f"Sample {j}")
                self.mask_indices = []
                loss = original_loss.detach().clone()
                for i in np.random.permutation(n):
                    self.mask_indices.append(i)
                    new_loss = self.run_all_forward()
                    sv[:, i] += ((new_loss - loss) / samples).squeeze().detach().cpu().numpy()
                    loss = new_loss

            # zero_player_loss = self.run_forward(loss_reduction="none")
            # print ((zero_player_loss-original_loss).mean(0))
            # print (sv.mean(0).sum())
            handle.remove()
            return self.aggregate_over_samples(sv)

    def _forward_hook(self):
        def _hook(module, _, output):
            return output.index_fill_(
                1, torch.tensor(self.mask_indices).to(self.device), 0.0,
            )

        return _hook
