from cnnlib import DataUtility
from albumentations import Compose
from albumentations.pytorch import *
from cnnlib import DataUtility
from cnnlib.DataUtility import Alb
from cnnlib.models import Resnet
from torchsummary import summary
from cnnlib import Utility
import torch.optim as optim
from cnnlib.ModelBuilder import ModelBuilder
from cnnlib.Functions import LossFn
import torch.nn as nn


def main():
    transforms = Compose([
        ToTensor()
    ])

    data = DataUtility.loadTinyImagenet("data/tiny-imagenet-200", Alb(transforms), Alb(transforms))
    DataUtility.showLoaderImages(data.test, classes=data.classes)


if __name__ == "__main__":
    main()
