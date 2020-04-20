import torch
from torchvision import datasets
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import csv
from cnnlib import Utility
from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from os.path import isfile, join
from cnnlib.datasets import ImageNet
from tqdm.autonotebook import tqdm as tqdm


class Data:
    """
    Bundles train, test loaders with index to class mappings if required.
    """

    def __init__(self, train_loader, test_loader, classes=None):
        self.train = train_loader
        self.test = test_loader
        self.classes = classes


def shape(loader):
    d, l = iter(loader).next()
    return d.shape


def download_CIFAR10(train_transforms, test_transforms, batch_size=128, isCuda=Utility.isCuda()):
    """
        Load CIFAR10 dataset. Uses the provided train_transforms and the test_transforms and create a object of Data.

        :param train_transforms: Transfomrations for train
        :param test_transforms: Transformations for test
        :param batch_size: Default value is 128
        :param isCuda: Default value is True
        :return: Data
        """
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if isCuda else dict(
        shuffle=True, batch_size=batch_size)

    train_data = datasets.CIFAR10("../data", train=True, transform=train_transforms, download=True)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    test_data = datasets.CIFAR10("../data", train=False, transform=test_transforms, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    print(f'Shape of a train data batch: {shape(train_loader)}')
    print(f'Shape of a test data batch: {shape(test_loader)}')

    print(f'Number of train images: {len(train_data.data)}')
    print(f'Number of test images: {len(test_data.data)}')

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return Data(train_loader, test_loader, classes)


def showImages(images, targets, predictions=None, cols=10, figSize=(15, 15)):
    """
    Shows images with its labels. Expected numpy arrays for images and the labels.
    """
    figure = plt.figure(figsize=figSize)
    num_of_images = len(images)
    rows = np.ceil(num_of_images / float(cols))
    for index in range(0, num_of_images):
        plt.subplot(rows, cols, index + 1)
        plt.axis('off')
        plt.imshow(images[index].squeeze())
        if predictions is None:
            plt.title(f"Tru={targets[index]}")
        else:
            plt.title(f"Tru={targets[index]}, Pred={predictions[index]}")


def showLoaderImages(loader, classes=None, count=20, muSigmaPair=None):
    """

    Takes random images from the loader and shows the images.
    Optionally Mean and Sigma pair can be passed to unnormalize data before showing the image.

    :param muSigmaPair: Default is (0, 1)
    """
    d, l = iter(loader).next()

    randImages = Utility.pickRandomElements(d, count)
    images = d[randImages]

    if (muSigmaPair is not None):
        images = Utility.unnormalize(images, muSigmaPair[0], muSigmaPair[1])

    # Loader has the channel at 1 index. But the show images need channel at the end.
    images = images.permute(0, 2, 3, 1)
    labels = __getLabels(l, randImages, classes)
    showImages(images.numpy(), labels, cols=5)


def showRandomImages(data, targets, predictions=None, classes=None, count=20, muSigmaPair=None):
    randImages = Utility.pickRandomElements(data, count)
    images = data[randImages]

    if (muSigmaPair is not None):
        images = Utility.unnormalize(images, muSigmaPair[0], muSigmaPair[1])

    images = images.permute(0, 2, 3, 1)

    targets = __getLabels(targets, randImages, classes)
    if predictions is not None:
        predictions = __getLabels(predictions, randImages, classes)

    showImages(images.numpy(), targets, predictions, cols=5)


def loadTinyImagenet(data_folder, train_transforms, test_transforms, batch_size=128, isCuda=Utility.isCuda()):
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if isCuda else dict(
        shuffle=True, batch_size=batch_size)

    train_data = ImageNet.TinyImageNet(data_folder, train=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)

    test_data = ImageNet.TinyImageNet(data_folder, train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)

    print(f'Shape of a train data batch: {shape(train_loader)}')
    print(f'Shape of a test data batch: {shape(test_loader)}')

    print(f'Number of train images: {len(train_data)}')
    print(f'Number of test images: {len(test_data)}')

    return Data(train_loader, test_loader, train_data.idx_class)


def computeMeanAndStd(train_loader, test_loader=None):
    data = []

    for images, _ in tqdm(train_loader):
        data.append(images)

    if test_loader:
        for images, _ in tqdm(test_loader):
            data.append(images)

    data = torch.cat(data)
    mean = data.mean(dim=0).mean(dim=1).mean(dim=1)
    std = data.std(dim=0).std(dim=1).std(dim=1)

    return mean, std


def __getLabels(labels, randoms, classes):
    labels = labels[randoms].numpy()
    if classes != None:
        labels = np.array([classes[i] for i in labels])

    return labels


class Alb:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img = np.array(img)
        img = self.transforms(image=img)['image']
        return img
