from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(28 * 28, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 10),
        )

    def forward(self, x):
        return self.fc(x)

    def forward_partial(self, x, from_module=None, to_module=None):
        processing = False
        for m in self.fc.children():
            if m == from_module:
                processing = True
                continue
            if from_module is None or processing:
                x = m(x)
                if m == to_module:
                    return x
        return x


def get_fc_model_with_name():
    model = Net()
    return model, "MNIST"


def loss(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def get_optimizer_for_model(model):
    return optim.SGD(model.parameters(), lr=0.01), None

def get_dataset_and_loaders(use_cuda=torch.cuda.is_available(), val_split=1000, val_batch_size=1000):
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_set = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(),]),
    )

    test_set = datasets.MNIST(
        "../data", train=False, transform=transforms.Compose([transforms.ToTensor(),]),
    )

    train_set, val_set = torch.utils.data.random_split(
        train_set, [len(train_set) - val_split, val_split]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=True, **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=500, shuffle=True, **kwargs,
    )

    return train_loader, validation_loader, test_loader
