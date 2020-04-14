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


class ImageNetLikeNIPS2017(Dataset):
    base_folder = 'NeurIPS2017Adversarial/'
    #TODO: change link
    url = 'https://drive.google.com/uc?export=download&confirm=_acC&id=15GmSKu0JChabYpT3NQ4XQTdNswSiq_aw'
    filename = 'NeurIPS2017Adversarial.tar.gz'
    tgz_md5 = 'f3189834df8758b62a9f6729d3d397f8'

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
        print(os.path.join(self.root, self.base_folder, 'dev_dataset.csv'))
        images = pd.read_csv(os.path.join(self.root, self.base_folder, 'dev_dataset.csv'))
        self.data = images
        print (self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, "images", f"{row.ImageId}.png")
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, None)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root+self.base_folder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.shared_dict:
            sample = self.data.iloc[idx]
            path = os.path.join(self.root, self.base_folder, "images", f"{sample.ImageId}.png")
            target = sample.TrueLabel
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)

            self.shared_dict[idx] = (img, target)

        return self.shared_dict[idx]


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



def prunable_vgg16(num_classes=1000):
    """Constructs a VGG-16"""
    model = vgg16_bn(num_classes=num_classes, pretrained=True)
    VGG.forward = vgg_forward_partial
    VGG.forward_partial = vgg_forward_partial
    return model


def get_vgg_model_with_name():
    model = prunable_vgg16()
    return model, "ImageNet-VGG16"


def loss(output, target, reduction="mean"):
    return F.cross_entropy(output, target, reduction=reduction)


def get_optimizer_for_model(model):
    opt = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        opt, milestones=[30, 60, 90, 120, 150], gamma=0.5
    )
    return opt, scheduler


def get_dataset():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    dataset = ImageNetLikeNIPS2017(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return dataset


def get_dataset_and_loaders(
    use_cuda=torch.cuda.is_available(),
    val_split=1000,
    val_from_test=False,
    val_batch_size=100,
):

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    dataset = get_dataset()

    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset) - 200, 200]
    )

    train_set, val_set = torch.utils.data.random_split(
        train_set, [len(train_set) - 600, 200]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=50, shuffle=True, **kwargs,
    )

    validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=50, shuffle=False, **kwargs,
    )

    return train_loader, validation_loader, test_loader


