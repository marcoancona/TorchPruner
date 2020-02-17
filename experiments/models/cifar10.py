from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16_bn, VGG


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32*32*3, 2024),
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


def vgg_forward_partial(self, x, to_module=None, from_module=None):
    processing = from_module is None

    for module in self.features.children():
        if processing:
            x = module(x)
            if module == to_module:
                return x
        elif module == from_module:
            processing = True

    x = torch.flatten(x, 1)

    for module in self.classifier.children():
        if processing:
            x = module(x)
            if module == to_module:
                return x
        elif module == from_module:
            processing = True
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
        elif (
            any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]])
            and current is not None
        ):
            pruning[-1][1].append(module)
    return pruning[::-1][1:]


def prunable_vgg16(num_classes=10):
    """Constructs a VGG-11 model for CIFAR dataset"""
    model = vgg16_bn(num_classes=num_classes)
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Linear(512, 10),
    )
    model._initialize_weights()
    VGG.forward = vgg_forward_partial
    VGG.forward_partial = vgg_forward_partial
    return model


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 32 * 3, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 2024),
            nn.LeakyReLU(),
            nn.Linear(2024, 10),
        )


def get_vgg_model_with_name():
    model = prunable_vgg16()
    return model, "CIFAR10-VGG16"


def get_fc_model_with_name():
    model = Net()
    return model, "CIFAR10-FC"


def loss(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def get_optimizer_for_model(model):
    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 90, 120, 150], gamma=0.5)
    return opt, scheduler


def get_dataset(use_cuda=torch.cuda.is_available()):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_set = datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_set = datasets.CIFAR10(
        "../data",
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    return train_set, test_set

def get_dataset_and_loaders(
    use_cuda=torch.cuda.is_available(), val_split=1000, val_from_test=False, val_batch_size=100
):

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    train_set, test_set = get_dataset()

    if val_from_test is True:
        test_set, val_set = torch.utils.data.random_split(
            test_set, [len(test_set) - val_split, val_split]
        )
    else:
        train_set, val_set = torch.utils.data.random_split(
            train_set, [len(train_set) - val_split, val_split]
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=50, shuffle=True, **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=250, shuffle=False, **kwargs,
    )

    return train_loader, validation_loader, test_loader
