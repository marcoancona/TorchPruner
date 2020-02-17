from unittest import TestCase
import pkg_resources
import logging, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F


from torchpruner.attributions import (
    WeightNormAttributionMetric,
    RandomAttributionMetric,
    SensitivityAttributionMetric,
    TaylorAttributionMetric,
    APoZAttributionMetric,
    ShapleyAttributionMetric
)


def max_model(device, version=1):
    # Make sure symmetric inputs are provided
    x = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
    y = np.array([[np.max(xi)] for xi in x])
    x = torch.tensor(x).float().to(device)
    y = torch.tensor(y).float().to(device)

    if version == 1:
        # Perfect solution
        w1 = torch.tensor(
            np.array([[-0.5, 1.0, 1.0, 1.0], [0.5, -1.0, 1.0, 1.0]])
        ).float()
        w2 = torch.tensor(np.array([[1], [0.5], [0.5], [0.0]])).float()
    elif version == 2:
        # Perfect solution except unit (D) which has a non-zero outgoing edge
        w1 = torch.tensor(
            np.array([[-0.5, 1.0, 1.0, 1.0], [0.5, -1.0, 1.0, 1.0]])
        ).float()
        w2 = torch.tensor(np.array([[1], [0.5], [0.5,], [-0.1]])).float()

    linear1 = nn.Linear(2, 4, bias=False)
    linear1.weight.data = torch.t(w1).to(device)
    linear2 = nn.Linear(4, 2, bias=False)
    linear2.weight.data = torch.t(w2).to(device)

    model = nn.Sequential(linear1, nn.ReLU(), linear2).to(device)
    return x, y, model



class TestTorchPruner(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        pass

    def test_max_model(self):
        x, y, model = max_model(self.device)
        y_pred = model(x)
        np.testing.assert_array_almost_equal(
            y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
        )

    def test_random(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = RandomAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])

    def test_weight_norm(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = WeightNormAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [1, 2, 2, 2])

    def test_apoz(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = APoZAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.5, 0.5, 1, 1])

    def test_sensitivity(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = SensitivityAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.0, 0.0, 0.0, 0.0])

    def test_taylor(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = TaylorAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.0, 0.0, 0.0, 0.0])

    def test_sv(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = ShapleyAttributionMetric(model, datagen, F.mse_loss, self.device, sv_samples=1000)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.37, 0.37, 1.7, 0.], decimal=1)

    def test_sensitivity_2(self):
        x, y, model = max_model(self.device, version=2)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = SensitivityAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        # (A) has double the weight of (B)
        # (B) is not active for half of the times so must have half gradient than (C) with the same weight
        # (D) is active as (C) but has a gradient 5 times smaller
        np.testing.assert_array_almost_equal(attr, [0.2, 0.1, 0.2, 0.04])

    def test_taylor_2(self):
        x, y, model = max_model(self.device, version=2)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = TaylorAttributionMetric(model, datagen, F.mse_loss, self.device)

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.1, 0.1, 0.5, 0.1])

    def test_taylor_2_signed(self):
        x, y, model = max_model(self.device, version=2)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        a = TaylorAttributionMetric(
            model, datagen, F.mse_loss, self.device, signed=True
        )

        attr = a.run(list(model.children())[0])
        self.assertEqual(list(attr.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.1, 0.1, 0.5, -0.1])

    def test_find_best_evaluation_module(self):
        """
        Given a Linear of Conv layer, we want to compute attributions *after* any BatchNorm
        or activation layers that might follow.
        """
        x, y, _ = max_model(self.device, version=2)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )
        # Define model with BatchNorm and ReLU
        model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 1)).to(
            self.device
        )

        for A in [TaylorAttributionMetric, SensitivityAttributionMetric, ShapleyAttributionMetric, APoZAttributionMetric]:
            # For these methods, should skip BN and ReLU
            a = A(model, datagen, F.mse_loss, self.device)
            eval_module = a.find_evaluation_module(list(model.children())[0], find_best_evaluation_module=True)
            self.assertEqual(eval_module, list(model.children())[2])

        for A in [WeightNormAttributionMetric, RandomAttributionMetric]:
            # For these methods, should NOT skip BN and ReLU
            a = A(model, datagen, F.mse_loss, self.device)
            eval_module = a.find_evaluation_module(list(model.children())[0], find_best_evaluation_module=True)
            self.assertEqual(eval_module, list(model.children())[0])

    def test_run_all_with_find_best_evaluation_module(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )

        for A in [TaylorAttributionMetric, SensitivityAttributionMetric,
                  APoZAttributionMetric, WeightNormAttributionMetric]:
            # For these methods, attributions before and after a ReLU should be the same
            # (Also for SV and Random, but these are not deterministic)
            a = A(model, datagen, F.mse_loss, self.device)
            attr = a.run(list(model.children())[0], find_best_evaluation_module=False)
            attr_best = a.run(list(model.children())[0], find_best_evaluation_module=True)
            np.testing.assert_array_almost_equal(attr, attr_best)

    def test_run_all_with_find_best_evaluation_module_2(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            dataset=TensorDataset(x, y), batch_size=1, shuffle=False,
        )

        for A in [TaylorAttributionMetric, SensitivityAttributionMetric,
                  APoZAttributionMetric, WeightNormAttributionMetric, RandomAttributionMetric, ShapleyAttributionMetric]:
            # Check if all methods run with find_best_evaluation_module = True
            a = A(model, datagen, F.mse_loss, self.device)
            attr_best = a.run(list(model.children())[0], find_best_evaluation_module=True)
            self.assertEqual(list(attr_best.shape), [4])
