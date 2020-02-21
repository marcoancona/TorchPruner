import torch
import numpy as np
import logging
from ..attributions import _AttributionMetric


class ShapleyAttributionMetric(_AttributionMetric):
    """
    Compute attributions as approximate Shapley values using sampling.
    """

    def __init__(self, *args, sv_samples=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = sv_samples
        self.mask_indices = []

    def run(self, module, sv_samples=None, **kwargs):
        module = super().run(module, **kwargs)
        sv_samples = sv_samples if sv_samples is not None else self.samples
        if hasattr(self.model, "forward_partial"):
            result = self.run_module_with_partial(module, sv_samples)
        else:
            logging.warning("Consider adding a 'forward_partial' method to your model to speed-up Shapley values "
                            "computation")
            result = self.run_module(module, sv_samples)
        return result

    def run_module_with_partial(self, module, sv_samples):
        """
        Implementation of Shapley value monte carlo sampling for models
        that provides a `forward_partial` function. This is significantly faster
        than run_module(), as it only runs the forward pass on the necessary modules.
        """
        d = len(self.data_gen.dataset)
        sv = None
        permutations = None
        c = 0

        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data_gen):
                x, y = x.to(self.device), y.to(self.device)
                original_z, _ = self.run_forward_partial(x, to_module=module)
                _, original_loss = self.run_forward_partial(original_z, y_true=y, from_module=module)
                n = original_z.shape[1]  # prunable dimension
                if permutations is None:
                    # Keep the same permutations for all batches
                    permutations = [np.random.permutation(n) for _ in range(sv_samples)]
                if sv is None:
                    sv = np.zeros((d, n))

                for j in range(sv_samples):
                    loss = original_loss.detach().clone()
                    z = original_z.clone().detach()

                    for i in permutations[j]:
                        z.index_fill_(1, torch.tensor(np.array([i])).long().to(self.device), 0.0)
                        _, new_loss = self.run_forward_partial(z, y_true=y, from_module=module)
                        delta = new_loss - loss
                        n = delta.shape[0]
                        sv[c:c+n, i] += (delta / sv_samples).squeeze().detach().cpu().numpy()
                        loss = new_loss
                c += n

            return self.aggregate_over_samples(sv)

    def run_module(self, module, samples):
        """
        Implementation of Shapley value monte carlo sampling.
        No further changes to the model are necessary but this can be quite slow.
        See run_module_with_partial() for a faster version that uses partial evaluation.
        """
        with torch.no_grad():
            self.mask_indices = []
            handle = module.register_forward_hook(self._forward_hook())
            original_loss = self.run_all_forward()
            n = module._tp_prune_dim  # output dimension
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

            handle.remove()
            return self.aggregate_over_samples(sv)

    def _forward_hook(self):
        def _hook(module, _, output):
            module._tp_prune_dim = output.shape[1]
            return output.index_fill_(
                1, torch.tensor(self.mask_indices).long().to(self.device), 0.0,
            )

        return _hook
