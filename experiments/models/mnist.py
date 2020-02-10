from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 60, kernel_size=(5, 5), stride=1, padding=2)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(60, 160, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(160, 1200, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(1200, 840)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(840, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

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

        for module in self.convnet.children():
            if evaluate:
                x = forward_module(module, x)
                if module == return_intermediate_output_module:
                    return x
            elif module == process_as_intermediate_output_module:
                evaluate = True

        x = torch.flatten(x, 1)

        for module in self.fc.children():
            if evaluate:
                x = forward_module(module, x)
                if module == return_intermediate_output_module:
                    return x
            elif module == process_as_intermediate_output_module:
                evaluate = True

        return x

    def get_pruning_graph(self):
        modules = list(self.convnet.children()) + list(self.fc.children())
        pruning = []
        current = None

        for module in modules:
            if any([isinstance(module, c) for c in [nn.Linear, nn.Conv2d]]):
                if current is not None:
                    pruning[-1][1].append(module)
                    pruning[-1][1].reverse()
                current = module
                pruning.append((module, []))
            elif any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]]) and current is not None:
                pruning[-1][1].append(module)
        return pruning[::-1][1:]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(28*28, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 10)
        )


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

        for module in self.fc.children():
            if evaluate:
                x = forward_module(module, x)
                if module == return_intermediate_output_module:
                    return x
            elif module == process_as_intermediate_output_module:
                evaluate = True

        return x

    def get_pruning_graph(self):
        modules = list(self.fc.children())
        pruning = []
        current = None

        for module in modules:
            if any([isinstance(module, c) for c in [nn.Linear, nn.Conv2d]]):
                if current is not None:
                    pruning[-1][1].append(module)
                    pruning[-1][1].reverse()
                current = module
                pruning.append((module, []))
            elif any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]]) and current is not None:
                pruning[-1][1].append(module)
        return pruning[::-1][1:]


def get_model_with_name():
    model = Net()
    return model, "MNIST"


def loss(output, target, reduction = "mean"):
    return F.cross_entropy(output, target, reduction = reduction)


def get_optimizer_for_model(model, epoch, prev_state=None):
    lr = 0.01 * (0.5 ** (epoch // 15))
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if prev_state is not None:
        opt.load_state_dict(prev_state)
        # Set again new lr, because it was replaced by load_state_dict
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return opt

def get_dataset_and_loaders(use_cuda=torch.cuda.is_available()):

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    dataset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    testset = datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )

    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - 10000, 10000])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=100,
        shuffle=True,
        **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=10000,
        shuffle=False,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=500,
        shuffle=True,
        **kwargs,
    )

    return dataset, train_loader, validation_loader, test_loader



