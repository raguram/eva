from albumentations import *
from albumentations.pytorch import ToTensor
from cnnlib import Utility
from cnnlib.DataUtility import Alb
import torch.optim as optim
from cnnlib.datasets.DepthDataset import DepthDataset
from cnnlib.models.ResUNet import ResUNet_Lite
import torch
from torchsummary import summary
from torch.nn import BCEWithLogitsLoss
from cnnlib.image_seg.ModelBuilder import ModelBuilder
from cnnlib.DataUtility import Data
from cnnlib.image_seg.PredictionPersister import ZipPredictionPersister

transforms = Alb(Compose([
    ToTensor()
]))

dataset = DepthDataset("data/tiny_data/", transforms, transforms, transforms, transforms)

train_dataset = torch.utils.data.Subset(dataset, list(range(16)))
test_dataset = torch.utils.data.Subset(dataset, list(range(16, 20)))

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=2)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=2)

model = ResUNet_Lite(6, 1)
summary(model, (6, 224, 224))

loss_fn = BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
persister = ZipPredictionPersister(zip_file_name="test-output.zip")

builder = ModelBuilder(model=model, data=Data(train_loader=train_loader, test_loader=test_loader), loss_fn=loss_fn,
                       optimizer=optimizer, train_pred_persister=persister, test_pred_persister=persister,
                       checkpoint=1, model_path="model.pt")

result = builder.fit(1)
print(result.train_losses, result.test_losses)
