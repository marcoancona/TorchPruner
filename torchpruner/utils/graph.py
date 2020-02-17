import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.activation import ReLU, ReLU6, RReLU, LeakyReLU, Sigmoid, Softplus, Tanh
import logging

ACTIVATIONS = [ReLU, ReLU6, RReLU, LeakyReLU, Sigmoid, Softplus, Tanh]


def find_best_module_for_attributions(model, module):
    """
    Given a Linear or Convolutional module, this method checks if the module
    if followed by either BatchNormalization and/or a non-linearity.
    In this case, returns the last of these modules as better location
    to compute attributions on.
    :param model: PyTorch module containing module
    :param module: PyTorch Linear or Conv module
    :return:
    """
    modules = list(model.modules())
    try:
        current_idx = modules.index(module)
        eval_module = module
        for next_module in modules[current_idx+1:]:
            if isinstance(next_module, _BatchNorm):
                print(f"BatchNorm detected: shifting evaluation after {next_module}")
                eval_module = next_module
            elif any([isinstance(next_module, t) for t in ACTIVATIONS]):
                print(f"Activation detected: shifting evaluation after {next_module}")
                eval_module = next_module
            else:
                return eval_module
    except ValueError:
        logging.error("Provided module is not in model")
    return module


def get_vgg_pruning_graph(vgg):
    """
    Returns a list of tuples (module, [cascading_modules])
    where module is a prunable 'module' and cascading_modules contains all
    the modules that should be pruned in consequence of 'module' being pruned
    :param vgg: PyTorch vgg model
    :return:
    """
    modules = list(vgg.features.children()) + list(vgg.classifier.children())
    pruning = []
    current = None

    for module in modules:
        if any([isinstance(module, c) for c in [nn.Linear, nn.Conv2d]]):
            if current is not None:
                pruning[-1][1].append(module)
                pruning[-1][1].reverse()
            current = module
            pruning.append((module, []))
        elif (
            any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]])
            and current is not None
        ):
            pruning[-1][1].append(module)
    return pruning[::-1][1:]