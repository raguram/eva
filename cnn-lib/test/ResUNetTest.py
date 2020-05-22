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
from cnnlib.models.ResUNet import ResUNet
from cnnlib.ModelBuilder import PredictionResult
import torch
from torchsummary import summary
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

print(torch.__version__)


class ModelTrainer:

    def __train_one_batch(self, model, data, target, optimizer, lossFn):
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(dim=1)
        loss = lossFn(output, target)
        loss.backward()
        optimizer.step()
        return (loss, target)

    def train_one_epoch(self, model, train_loader, optimizer, device, lossFn, scheduler):
        model.train()
        pbar = tqdm(train_loader)
        whole_target = []
        total_loss = 0
        for idx, data in enumerate(pbar):
            x = torch.cat((data['bg'], data['fg_bg']), dim=1)
            (loss, target) = self.__train_one_batch(model, x, data['fg_bg_mask'], optimizer, lossFn)
            total_loss += loss
            whole_target.append(target)
            break

        return total_loss, torch.cat(whole_target)


transforms = Alb(Compose([
    ToTensor()
]))

dataset = DepthDataset("data/tiny_data/", transforms, transforms, transforms, transforms)
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=2)

trainer = ModelTrainer()
model = ResUNet(6, 1).to(Utility.getDevice())
summary(model, (6, 224, 224))

optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

lossFn = BCEWithLogitsLoss()

loss, whole_target = trainer.train_one_epoch(model=model, train_loader=loader, optimizer=optimizer,
                                             device=Utility.getDevice(),
                                             lossFn=lossFn, scheduler=optimizer)

print(loss)