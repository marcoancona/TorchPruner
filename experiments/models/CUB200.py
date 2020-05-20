from __future__ import print_function
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.models import vgg11, vgg16_bn, vgg16, VGG
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd
import multiprocessing as mp
import ctypes


class CUB2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        self.shared_dict = {}

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.shared_dict:
            sample = self.data.iloc[idx]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)

            self.shared_dict[idx] = (img, target)

        return self.shared_dict[idx]


def forward(
    self,
    x,
    return_intermediate_output_module=None,
    process_as_intermediate_output_module=None,
    linearize=False,
):

    evaluate = process_as_intermediate_output_module is None

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
        elif (
            any([isinstance(module, c) for c in [nn.BatchNorm2d, nn.Dropout]])
            and current is not None
        ):
            pruning[-1][1].append(module)
    return pruning[::-1][1:]

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





def prunable_vgg16(num_classes=200):
    model = vgg16_bn(pretrained=False)
#     num_ftrs = model.classifier[6].in_features
#     model.classifier[6] = nn.Linear(num_ftrs, 200)
    model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    model._initialize_weights()
    VGG.forward = vgg_forward_partial
    VGG.forward_partial = vgg_forward_partial
    return model


def get_model_with_name():
    model = prunable_vgg16()
    return model, "CUB200-VGG16"


def loss(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def get_optimizer_for_model(model, epoch, prev_state=None, lr=0.001):
    print("Learning rate: ", lr)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    if prev_state is not None:
        opt.load_state_dict(prev_state)
        # Set again new lr, because it was replaced by load_state_dict
        for param_group in opt.param_groups:
            param_group["lr"] = lr
    return opt


def get_dataset(use_cuda=torch.cuda.is_available()):
    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    dataset_augmented = CUB2011(
        "../data",
        download=True,
        train=True,
        transform=data_transforms['train']
    )

    train_set, test_set = torch.utils.data.random_split(dataset_augmented, [len(dataset_augmented) - 1000, 1000])
    return train_set, test_set


def get_dataset_and_loaders(use_cuda=torch.cuda.is_available()):

    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
    }

    dataset_augmented = CUB2011(
        "../data",
        download=True,
        train=True,
        transform=data_transforms['train']
    )

    _, val_set = torch.utils.data.random_split(dataset_augmented, [len(dataset_augmented) - 1000, 1000])

    train_loader = torch.utils.data.DataLoader(
        dataset_augmented,
        batch_size=64,
        shuffle=True,
        **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=50,
        shuffle=False,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        CUB2011(
            "../data",
            transform=data_transforms['test'],
            train=False
        ),
        batch_size=64,
        shuffle=True,

        **kwargs,
    )

    return dataset_augmented, train_loader, validation_loader, test_loader


def test():
    from torchsummary import summary
    from thop import profile
    from shapley_pruning.prunable import ContinuousPruner

    model, name = get_model_with_name()
    dataset, train_loader, validation_loader, test_loader = get_dataset_and_loaders(use_cuda=False)

    x, y_true = next(iter(train_loader))
    print (x.shape)
    print (y_true)
    print(len(train_loader.dataset))

if __name__ == "__main__":
    test()

