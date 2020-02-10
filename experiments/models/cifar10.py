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
        elif any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]]) and current is not None:
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
    VGG.forward = forward
    VGG.get_pruning_graph = get_pruning_graph
    return model


class SimpleFCNet(nn.Module):
    def __init__(self):
        super(SimpleFCNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 32 * 3, 2024),
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
    model = prunable_vgg16()
    return model, "CIFAR10-VGG16"
    # model = SimpleFCNet()
    # return model, "CIFAR10-FC"


def loss(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def get_optimizer_for_model(model, epoch, prev_state=None):
    lr = 0.05 * (0.5 ** (epoch // 30))
    lr = 0.005
    print('Learning rate: ', lr)
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if prev_state is not None:
        opt.load_state_dict(prev_state)
        # Set again new lr, because it was replaced by load_state_dict
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return opt



def get_dataset_and_loaders(use_cuda=torch.cuda.is_available()):

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_augmented = datasets.CIFAR10(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
    )

    testset = datasets.CIFAR10(
            "../data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        )

    test_set, reduced_train_set = torch.utils.data.random_split(testset, [len(testset) - 1000, 1000])
    reduced_train_set_prune, reduced_train_set_val = torch.utils.data.random_split(reduced_train_set, [len(reduced_train_set) - 500, 500])

    train_loader = torch.utils.data.DataLoader(
        reduced_train_set_prune,
        batch_size=50,
        shuffle=True,
        **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        reduced_train_set_val,
        batch_size=50,
        shuffle=False,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=250,
        shuffle=False,
        **kwargs,
    )

    return dataset_augmented, train_loader, validation_loader, test_loader


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

    state = opt.state_dict()
    pg = model.get_pruning_graph()
    pruner = ContinuousPruner(model, (3, 32, 32), pg, 'cpu', validation_loader, test_loader, loss, verbose=1)
    pruner.prune(0.5, prune_all_layers=True, optimizer=opt, ranking_method='random')


if __name__ == "__main__":
    test()
