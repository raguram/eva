from albumentations import *
from albumentations.pytorch import ToTensor
from cnnlib import Utility
from cnnlib.models.S11Resnet import S11Resnet
from torch.nn import functional as F
from cnnlib.DataUtility import Alb
import numpy as np
from cnnlib import DataUtility
import torch.optim as optim
from cnnlib.lr_finder import LRFinder
from cnnlib.Functions import LossFn
from cnnlib.ModelBuilder import ModelBuilder
import torch.nn as nn

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

mean_array = np.array([*mean])

train_transforms = Compose([
    PadIfNeeded(40, 40, always_apply=True, p=1.0),
    RandomCrop(32, 32, always_apply=True, p=1.0),
    HorizontalFlip(p=0.5),
    Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=np.array([*mean]) * 255.0, p=0.75),
    Normalize(mean, std),
    ToTensor()
])

test_transforms = Compose([
    Normalize(mean, std),
    ToTensor()
])

data = DataUtility.download_CIFAR10(Alb(train_transforms), Alb(test_transforms), batch_size=512)

net = S11Resnet().to(Utility.getDevice())
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
builder = ModelBuilder(net, data, LossFn(F.nll_loss, l2Factor=0.01, model=net), optimizer)
result = builder.fit(1)
