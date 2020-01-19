from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg11, vgg16_bn, vgg16, VGG



def forward(self, x,
            return_intermediate_output_module=None,
            process_as_intermediate_output_module=None,
            linearize=False):

    evaluate = process_as_intermediate_output_module is None

    # if return_intermediate_output_module:
        # print ("Forward with return_intermediate_output_module")

    def forward_module(module, x):
        # print(f".. forward {module}")
        if linearize:
            if any([isinstance(module, c) for c in [nn.ReLU]]):
                # print(f".. linearize")
                return x
            elif any([isinstance(module, c) for c in [nn.MaxPool2d]]):
                # print(f".. linearize")
                return F.avg_pool2d(x, module.kernel_size, module.stride)
        return module(x)

    for module in self.features.children():
        if evaluate:
            x = forward_module(module, x)
            if module == return_intermediate_output_module:
                return x
        elif module == process_as_intermediate_output_module:
            evaluate = True

    x = torch.flatten(x, 1)

    for module in self.classifier.children():
        if evaluate:
            x = forward_module(module, x)
            if module == return_intermediate_output_module:
                return x
        elif module == process_as_intermediate_output_module:
            evaluate = True

    return x


def get_pruning_graph(self):
    modules = list(self.features.children()) + list(self.classifier.children())
    pruning = []
    current = None

    for module in modules:
        if any([isinstance(module, c) for c in [nn.Linear, nn.Conv2d]]):
            if current is not None:
                pruning[-1][1].append(module)
                pruning[-1][1].reverse()
            current = module
            pruning.append((module, []))
        elif any([isinstance(module, c) for c in [nn.BatchNorm2d]]) and current is not None:
            pruning[-1][1].append(module)
    return pruning[::-1][1:]


def prunable_vgg11(num_classes=10):
    """Constructs a VGG-11 model for CIFAR dataset"""
    model = vgg11(num_classes=num_classes)
    model.classifier[0] = nn.Linear(512, 4096)
    VGG.forward = forward
    VGG.get_pruning_graph = get_pruning_graph
    return model


def prunable_vgg16(num_classes=10):
    """Constructs a VGG-11 model for CIFAR dataset"""
    model = vgg16_bn(num_classes=num_classes)
    model.classifier = nn.Sequential(nn.Linear(512, 10))
    VGG.forward = forward
    VGG.get_pruning_graph = get_pruning_graph
    return model


def get_model_with_name():
    model = prunable_vgg16()
    return model, "CIFAR10-VGG16"


def loss(output, target, reduction = "mean"):
    return F.nll_loss(F.log_softmax(output, dim=1), target, reduction=reduction)


def get_optimizer_for_model(model, epoch):
    lr = 1e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


def get_dataset_and_loaders(use_cuda=torch.cuda.is_available()):

    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    dataset = datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=250,
        shuffle=True,
        **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=500,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "../data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        ),
        batch_size=250,
        shuffle=True,
        **kwargs,
    )

    return dataset, train_loader, validation_loader, test_loader


def test():
    from torchsummary import summary
    from thop import profile
    from shapley_pruning.prunable import ContinuousPruner

    model, name = get_model_with_name()
    dataset, train_loader, validation_loader, test_loader = get_dataset_and_loaders(use_cuda=False)
    summary(model, input_size=(3, 32, 32), device="cpu")

    x = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(x,))
    print (macs)
    print (params)

    opt = get_optimizer_for_model(model, 0)
    pg = model.get_pruning_graph()
    pruner = ContinuousPruner(model, (3, 32, 32), pg, 'cpu', validation_loader, test_loader, loss, verbose=1)
    pruner.prune(0.5, prune_all_layers=True, optimizer=opt, ranking_method='random')


if __name__ == "__main__":
    test()
