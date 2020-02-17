# Torch Pruning
A library for structured pruning *on-the-fly* in PyTorch.

This library contains two main modules: 

- `attributions`: this module implements several attributions metrics (sometimes
also called pruning criteria) to evaluate the relevance of prunable units. 
In a classical pruning pipeline, these metrics can be used to identify
 the least relevant units in the network, which will be then pruned.

- `pruner`: this module provides tools to perform structured pruning of a PyTorch model. 
While pruning is often simulated with masking, this library instead performs actual slicing operations
on the parameter Tensors of the model to reduce the number of FLOPs.
Pruning of Linear and Convolutional modules is supported. 
In the case of Linear modules, structured pruning consists in removing one or more output neurons. 
In the case of Conv modules, output filters are removed.
It is important to notice that the pruning of a module often requires to perform side pruning operations
to keep the shape of the tensors of the *following* layers compatible. This library provides
support for cascading pruning of the following layers as well as of the optimizer parameters.





