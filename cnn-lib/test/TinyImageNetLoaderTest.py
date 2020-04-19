from cnnlib import DataUtility
from albumentations import Compose
from albumentations.pytorch import *
from cnnlib import DataUtility
from cnnlib.DataUtility import Alb


# What am I going to do?
# 1. Load the train data set. Load the appropriate label files and come up with the classes list
# 2. Load the validation data set. For this, I will create 2 arrays one for data and another for labels. Will use it to create data loader.
# Goal of this first step is to create Data object.

def main():
    transforms = Compose([
        ToTensor()
    ])

    data = DataUtility.loadTinyImagenet("data/tiny-imagenet-200", Alb(transforms), Alb(transforms))
    DataUtility.showLoaderImages(data.train, classes=data.classes)


if __name__ == "__main__":
    main()
