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

criterion = F.nll_loss
net = S11Resnet().to(Utility.getDevice())
optimizer = optim.SGD(net.parameters(), lr = 1e-5, momentum=0.9)
finder = LRFinder(net, optimizer, criterion, Utility.getDevice())

finder.range_test(data.train, val_loader=data.test, start_lr=1e-5, end_lr=1e-4,
                     num_iter=2, step_mode="linear")
finder.plot()
finder.reset()