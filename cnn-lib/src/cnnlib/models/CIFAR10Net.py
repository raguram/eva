from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10Net(nn.Module):

  def __init__(self):
    super(CIFAR10Net, self).__init__()

    drop_out = 0.1

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, dilation=2), # RF - 5
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False), # RF - 7
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout2d(drop_out),
    ) # Output - 30

    self.transition1 = nn.Sequential(
        nn.MaxPool2d(2, 2), ## RF - 8
    ) # Output - 15

    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False), # RF - 12
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(drop_out),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0, bias=False), # RF - 16
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout2d(drop_out),
    ) # Output - 11

    self.transition2 = nn.Sequential(
        nn.MaxPool2d(2, 2), ## RF - 18
    ) # Output - 5

    self.depthwiseSeparable = nn.Sequential(
        # Depthwise
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups=64), # RF - 26
        # Pointwise
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), bias=False), # RF - 26
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Dropout2d(drop_out)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # RF - 34
    ) # Output - 5

    self.gap1 = nn.AdaptiveAvgPool2d(1) # RF - 50
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), bias=False) # RF - 50

  def forward(self, x):

    x = self.transition1(self.conv1(x))
    x = self.transition2(self.conv2(x))
    x = self.conv3(self.depthwiseSeparable(x))
    x = self.conv4(self.gap1(x))
    x = x.view(-1, 10)
    return F.log_softmax(x)

  def summarize(self, input):
    summary(self, input_size=input)

CIFAR10Net().summarize((3, 32, 32))