from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(4096, 10)
        self.my_modules = [
            self.conv1,
            self.relu1,
            self.maxpool1,
            self.conv2,
            self.relu2,
            self.maxpool2,
            nn.Flatten(1),
            self.fc1,
            self.relu3,
            self.fc2,
        ]

    def forward(
        self,
        x,
        return_intermediate_output_module=None,
        process_as_intermediate_output_module=None,
        linearize=False,
    ):

        i, j = 0, len(self.my_modules)

        if process_as_intermediate_output_module is not None:
            i = 1 + self.my_modules.index(process_as_intermediate_output_module)
        if return_intermediate_output_module is not None:
            j = 1 + self.my_modules.index(return_intermediate_output_module)

        for module in self.my_modules[i:j]:
            if linearize is True and module in [self.relu1, self.relu2, self.relu3]:
                continue
            if linearize is True and module in [self.maxpool1, self.maxpool2]:
                x = F.avg_pool2d(x, 2)
            else:
                x = module(x)
        return x

    def get_pruning_graph(self):
        return [
            (self.fc1, [self.fc2]),
            (self.conv2, [self.fc1]),
            (self.conv1, [self.conv2]),
        ]


def get_model_with_name():
    return Net(), "FMNIST"


def loss(output, target, reduction="mean"):
    return F.nll_loss(F.log_softmax(output, dim=1), target, reduction=reduction)


def get_optimizer_for_model(model):
    lr = 0.001
    opt = optim.SGD(model.parameters(), lr=lr)
    return opt


def get_dataset_and_loaders(use_cuda=torch.cuda.is_available()):

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    dataset = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        ),
    )

    testset = datasets.FashionMNIST(
        "../data", train=False, transform=transforms.Compose([transforms.ToTensor(),]),
    )

    _, dataset_val = torch.utils.data.random_split(dataset, [len(dataset) - 1000, 1000])

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=True, **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1000, shuffle=False, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=500, shuffle=False, **kwargs,
    )

    return dataset, train_loader, validation_loader, test_loader
