TorchPruner:  *on-the-fly* structured pruning in PyTorch.
[![Build Status](https://travis-ci.org/marcoancona/TorchPruner.svg?branch=master)](https://travis-ci.org/marcoancona/TorchPruner)
===

This library provides tools to perform structured pruning of a PyTorch model. This includes a module to compute the "relevance score"
or attributions for all activations of a specific layers, and helpers to run the pruning itself. 

*Why on-the-fly*? Because TorchPruner performs real pruning (slicing of the parameter tensors), 
not just masking, therefore produces models with lower inference and training cost. Moreover, 
TorchPruner can prune the parameters of multiple layers, as well as fix Dropout and optimizer state, 
enabling pruning and training without the need to load a new model. 

 

- [`Attributions module`](#attributions-module): this module implements several attributions metrics (sometimes
also called pruning criteria) to evaluate the relevance of prunable units. 
In a classical pruning pipeline, these metrics can be used to identify
 the least relevant units in the network, which will be then pruned.

- [`Pruner module`](#pruner-module): this module provides tools to perform structured pruning of a PyTorch model. 

Structured pruning of *Linear* and *Convolutional* modules is supported at the moment.
 
In the case of Linear modules, structured pruning consists in removing one or more output neurons. 
In the case of Conv modules, output filters are removed.
It is important to notice that the pruning of a module often requires to perform side pruning operations
to keep the shape of the tensors of the *following* layers compatible. This library provides
support for cascading pruning of the following layers as well as of the optimizer parameters.



TorchPruner Quickstart
===
## Installation
```unix
TODO
```

Notice that DeepExplain assumes you already have installed `PyTorch >= 1.3`.

Attribution module
===
Implements the following attribution methods:

- `RandomAttributionMetric`
- `APoZAttributionMetric` (https://arxiv.org/abs/1607.03250)
- `SensitivityAttributionMetric` (https://arxiv.org/abs/1812.10240)
- `TaylorAttributionMetric` (https://openreview.net/forum?id=SJGCiw5gl)
- `WeightNormAttributionMetric` (https://arxiv.org/abs/1608.08710)
- `ShapleyAttributionMetric` 

#### Usage

```python
from torchpruner.attributions import RandomAttributionMetric  # or any of the methods above

attr = RandomAttributionMetric(model, data_generator, criterion, device)
for module in model.children():
    if len(list(module.children())) == 0:  # leaf module
        scores = attr.run(module)
        print (scores)      

```

#### `__init__(model, data_generator, criterion, device)`


Parameter name  | Type | Description
---------------|------|------------
`model` | PyTorch model, required | PyTorch model to compute attributions for.
`data_generator` | ` torch.utils.data.DataLoader`, required | DataLoader to generate the data used to compute attributions
`criterion` | callable, required | Loss function of the model. Should accept `input`, `target` and `reduction` params.
`device` | `torch.device`, required | Should be the same device your model and data run on.

#### `scores = run(module, find_best_evaluation_module=False, **kwargs)`


Parameter name  | Type | Description
---------------|------|------------
`module` | PyTorch module, required | A leaf PyTorch module within your model to compute attributions for.
`scores` | `np.array` | Scores for the module activations. This is a one-dimensional array where the length equals `z.shape[1]` (`z` being the module activation Tensor).
`find_best_evaluation_module` | bool, optional | By default, attributions are computed on the output of `module`. Sometimes it is preferable to compute attributions after BatchNormalization and nn-linear activation  


Additional parameters (`**kwargs`) are method-specific: 

Parameter name  | Type | Method(s) | Description
---------------|------|------|------------
`sv_samples` | int, optional, default=`5` | `ShapleyAttributionMetric` | How many samples to use to estimate Shapley values. Notice that the method requires `sv_samples * z.shape[1]` evaluations of the model, with `z` being the `module` activation.
`signed` | signed, optional, default=`False` | `TaylorAttributionMetric` | When `true` does not compute the absolute value before aggregating over samples.


Pruner module
===
This module provides the `Pruner` class to perform on-the-fly pruning of a PyTorch model. Pruning can happen at parameter, module or model levels:
- to prune a *single parameter Tensor* in a module, use the `prune_parameter()` method;
- to prune a *module*, use the `prune_module()` method, which is aware of the parameters that needs to be pruned
- to prune a *model and all cascading modules* (most common case), use the `prune_model()` method. This method provide a high-level pruning API to prune multiple layers, adjust Dropout rates and fix the optimizer parameters if necessary.

#### Usage

```python
from torchpruner.pruner import Pruner

pruner = Pruner(model, input_size=(c, w, h), device=device, optimizer=optimizer)
pruner.prune_model(one_module, indices=indices_to_prune, cascading_modules=other_modules_to_be_pruned)   

```

#### `__init__(model, input_size, device, optimizer=None)`


Parameter name  | Type | Description
---------------|------|------------
`model` | PyTorch model, required | PyTorch model to compute attributions for.
`input_size` | Tuple, required | Shape of the input (without batch dimension). Necessary to generate fictitious input to test the model.
`device` | `torch.device`, required | Should be the same device your model and data run on.
`optimizer` | `torch.Optimizer`, optional | If provided, TorchPruner will adjust the state of the optimizer (e.g. the `momentum` Tensor) to be compatible with the new parameter shape. Only `SGD` supported at the moment.

#### `prune_model(module, indices, cascading_modules=None`


Parameter name  | Type | Description
---------------|------|------------
`module` | PyTorch module, required | The module to prune (`nn.Linear` or any `nn._ConvNd` supported).
`indices` | List[int] | List of indices of the `module` activations to prune (along dimension 1). Module parameters (weight, bias, etc.) will be pruned accordingly.
`cascading_modules` | List[`torch.module`], optional | A list of modules following `module` in the computational graph that needs to be pruned to process the new input shape. This can include instances of `nn.Linear`, `nn._ConvNd`, `nn._DropoutNd`, `nn._BatchNorm`.


Other methods:
####  `prune_module(module, indices, direction="out", original_len=None)`
Prune units identified by `indices` in `module`. By default, the method prunes the layer parameters in their output dimension. To prune the parameters in their incoming direction use `direction="in"`. For Dropout layers only, `original_len` should be set equal to the length of the original input (along dimension 1).

####  `prune_parameter(self, module, parameter_name, indices, axis=0)`
Prune a single parameter Tensor identified by its name (`parameter_name`) within a `module`. The parameter Tensor is sliced such that `indices` are removed along the `axis` dimension.


