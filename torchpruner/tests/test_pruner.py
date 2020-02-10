from unittest import TestCase
import pkg_resources
import logging, warnings
import numpy as np
import torch
import torch.nn as nn

from torchpruner import Pruner


def simple_model(device):
    x, y = torch.ones((10, 3)), torch.randint(0, 10, (10, 1))
    return (
        (x.to(device), y.to(device)),
        nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1)).to(device),
    )


class TestTorchPruner(TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        pass

    def test_tf_available(self):
        try:
            pkg_resources.require("torch>=1.2.0")
        except Exception:
            self.fail("PyTorch requirement not met")

    def test_simple_model(self):
        (x, y), model = simple_model(self.device)

        _y = model(x)
        self.assertEqual(list(_y.shape), list(y.shape))

    def test_prune_parameter(self):
        """
        Should prune a parameter within a module, without affecting the parameter id.
        """
        (x, y), model = simple_model(self.device)
        p = Pruner(model, input_size=(3,), device=self.device)

        module = list(model.children())[0]
        weight_id = id(module.weight)
        self.assertEqual(list(module.weight.data.shape), [2, 3])
        self.assertEqual(list(module.bias.data.shape), [2])

        p.prune_parameter(module, "weight", [0], axis=0)
        p.prune_parameter(module, "bias", [0], axis=0)

        self.assertEqual(list(module.weight.data.shape), [1, 3])
        self.assertEqual(list(module.bias.data.shape), [1])
        self.assertEqual(id(module.weight), weight_id)

        p.prune_parameter(module, "weight", [0], axis=1)
        self.assertEqual(list(module.weight.data.shape), [1, 2])

    def test_prune_module_linear(self):
        (x, y), model = simple_model(self.device)
        p = Pruner(model, input_size=(3,), device=self.device)

        module = list(model.children())[0]
        p.prune_module(module, [0], direction="out")

        self.assertEqual(list(module.weight.data.shape), [1, 3])

        p.prune_module(module, [0], direction="in")
        self.assertEqual(list(module.weight.data.shape), [1, 2])

    def test_nan_trick_linear_linear(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1)).to(
            self.device
        )
        p = Pruner(model, input_size=(3,), device=self.device)
        list(model.children())[0].register_forward_hook(p._nanify_hook([0]))
        list(model.children())[2].register_forward_hook(p._detect_nan_hook())
        p._run_forward()
        indices = getattr(list(model.children())[2], "_nan_indices")
        self.assertEqual(list(indices), [0])

    def test_nan_trick_conv2d_linear(self):
        model = nn.Sequential(
            nn.Conv2d(1, 3, 2), nn.ReLU(), nn.Flatten(), nn.Linear(12, 1)
        ).to(self.device)
        p = Pruner(model, input_size=(1, 3, 3,), device=self.device)
        list(model.children())[0].register_forward_hook(p._nanify_hook([0]))
        list(model.children())[3].register_forward_hook(p._detect_nan_hook())
        p._run_forward()
        indices = getattr(list(model.children())[3], "_nan_indices")
        self.assertEqual(list(indices), [0, 1, 2, 3])

    def test_nan_trick_conv2d_max_linear(self):
        model = nn.Sequential(
            nn.Conv2d(1, 3, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(3, 1),
        ).to(self.device)
        p = Pruner(model, input_size=(1, 3, 3,), device=self.device)
        list(model.children())[0].register_forward_hook(p._nanify_hook([0]))
        list(model.children())[4].register_forward_hook(p._detect_nan_hook())
        p._run_forward()
        indices = getattr(list(model.children())[4], "_nan_indices")
        self.assertEqual(list(indices), [0])

    def test_nan_trick_linear_bn_linear(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2), nn.Linear(2, 1)).to(
            self.device
        )
        p = Pruner(model, input_size=(3,), device=self.device)
        list(model.children())[0].register_forward_hook(p._nanify_hook([0]))
        list(model.children())[1].register_forward_hook(p._detect_nan_hook())
        list(model.children())[2].register_forward_hook(p._detect_nan_hook())
        p._run_forward()
        indices = getattr(list(model.children())[1], "_nan_indices")
        self.assertEqual(list(indices), [0])
        indices = getattr(list(model.children())[2], "_nan_indices")
        self.assertEqual(list(indices), [0])

    def test_prune_model_linear(self):
        (x, y), model = simple_model(self.device)
        p = Pruner(model, input_size=(3,), device=self.device)

        module = list(model.children())[0]
        next_module = list(model.children())[2]
        pruning_indices = [0]

        p.prune_model(module, pruning_indices, cascading_modules=[next_module])

        self.assertEqual(list(module.weight.data.shape), [1, 3])
        self.assertEqual(list(next_module.weight.data.shape), [1, 1])
        self.assertEqual(list(model(x).shape), list(y.shape))

    def test_prune_model_linear_auto_detect(self):
        (x, y), model = simple_model(self.device)
        p = Pruner(model, input_size=(3,), device=self.device)

        module = list(model.children())[0]
        next_module = list(model.children())[2]
        pruning_indices = [0]

        p.prune_model(module, pruning_indices)  # <- auto detect cascading modules

        # Pruning indices in input to next module should correspond
        self.assertEqual(list(module.weight.data.shape), [1, 3])
        self.assertEqual(list(next_module.weight.data.shape), [1, 1])
        self.assertEqual(list(model(x).shape), list(y.shape))

    def test_prune_model_linear_bn_auto_detect(self):
        (x, y), _ = simple_model(self.device)
        model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2), nn.Linear(2, 1)).to(
            self.device
        )
        p = Pruner(model, input_size=(3,), device=self.device)

        module = list(model.children())[0]
        bn_module = list(model.children())[1]
        lin_module = list(model.children())[2]
        pruning_indices = [0]

        p.prune_model(module, pruning_indices)  # <- auto detect cascading modules

        # Pruning indices in input to next module should correspond
        self.assertEqual(list(module.weight.data.shape), [1, 3])
        self.assertEqual(list(lin_module.weight.data.shape), [1, 1])
        self.assertEqual(list(bn_module.weight.data.shape), [1])
        self.assertEqual(list(bn_module.bias.data.shape), [1])
        self.assertEqual(list(bn_module.running_var.data.shape), [1])
        self.assertEqual(list(bn_module.running_mean.data.shape), [1])

        self.assertEqual(list(model(x).shape), list(y.shape))
