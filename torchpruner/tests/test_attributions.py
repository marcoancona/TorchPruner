from unittest import TestCase
import pkg_resources
import logging, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchpruner import WeightNormAttributionMetric, \
    RandomAttributionMetric, SensitivityAttributionMetric, TaylorAttributionMetric, APoZAttributionMetric


def max_model(device):
    x = np.array([
        [0, 1],
        [1, 0],
        [1, 2]
    ])
    y = np.array([[np.max(xi)] for xi in x])
    x = torch.tensor(x).float().to(device)
    y = torch.tensor(y).float().to(device)

    w1 = torch.tensor(np.array([[-0.25, 1.0, 1.0, 1.0], [0.25, -1.0, 1.0, 1.0]])).float()
    w2 = torch.tensor(np.array([[2], [0.5], [0.5, ], [0.0]])).float()

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
        np.testing.assert_array_almost_equal(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    def test_random(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            (x, y),
            batch_size=1,
            shuffle=False,
        )
        a = RandomAttributionMetric(model, datagen, F.mse_loss, self.device)

        result = a.run([list(model.children())[0]])
        self.assertEqual(list(result[0][0].shape), [4])
        self.assertEqual(list(result[0][1].shape), [4])

    def test_weight_norm(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            (x, y),
            batch_size=1,
            shuffle=False,
        )
        a = WeightNormAttributionMetric(model, datagen, F.mse_loss, self.device)

        result = a.run([list(model.children())[0]])
        attr, rank = result[0]
        self.assertEqual(list(attr.shape), [4])
        self.assertEqual(list(rank.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.5, 2, 2, 2])

    def test_apoz(self):
        x, y, model = max_model(self.device)
        datagen = torch.utils.data.DataLoader(
            (x, y),
            batch_size=1,
            shuffle=False,
        )
        print (next(iter(datagen)))
        a = APoZAttributionMetric(model, datagen, F.mse_loss, self.device)

        result = a.run([list(model.children())[0]])
        attr, rank = result[0]
        self.assertEqual(list(attr.shape), [4])
        self.assertEqual(list(rank.shape), [4])
        np.testing.assert_array_almost_equal(attr, [0.5, 0.5, 1, 1])


