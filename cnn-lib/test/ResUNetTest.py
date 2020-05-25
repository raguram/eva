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
from cnnlib.datasets.DepthDataset import DepthDataset
import torch
from cnnlib.models.ResUNet import ResUNet_Lite as ResUNet
from cnnlib.ModelBuilder import PredictionResult
import torch
from torchsummary import summary
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from cnnlib.image_seg.ModelBuilder import *
from cnnlib.DataUtility import Data
from cnnlib.image_seg.Loss import Loss_fn

from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])

dataset = DepthDataset("data/tiny_data/", trans, trans, trans, trans)
train_dataset = torch.utils.data.Subset(dataset, list(range(8)))
test_dataset = torch.utils.data.Subset(dataset, list(range(16, 20)))
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=2)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=2)

# dataset.show_images(5)

model = ResUNet(6, 1).to(Utility.getDevice())
summary(model, (6, 224, 224))

optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

lossFn = Loss_fn(BCEWithLogitsLoss(), BCEWithLogitsLoss(), 1, 1)

builder = ModelBuilder(model=model, optimizer=optimizer,
                       device=Utility.getDevice(),
                       loss_fn=lossFn, scheduler=optimizer, data=Data(train_loader, test_loader))

result = builder.fit(1)
print(result)
