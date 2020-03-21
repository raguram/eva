from torchsummary import summary
import torch.nn as nn

"""
Requirements as per Quiz 9

x1 = Input
x2 = Conv(x1)
x3 = Conv(x1 + x2)
x4 = MaxPooling(x1 + x2 + x3)
x5 = Conv(x4)
x6 = Conv(x4 + x5)
x7 = Conv(x4 + x5 + x6)
x8 = MaxPooling(x5 + x6 + x7)
x9 = Conv(x8)
x10 = Conv (x8 + x9)
x11 = Conv (x8 + x9 + x10)
x12 = GAP(x11)
x13 = FC(x12)
"""

class InputBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out):
        super(InputBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(drop_out)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0, bias=False)
        )

        self.shortcut = nn.Sequential()
        if(in_channels != out_channels):
          # Use 1x1 to change the size of the channel for shortcut. This will be summed up with output from other conv layers
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False)
          )

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.conv1(x)
        x = self.conv2(x1 + x2)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(drop_out)

        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(drop_out)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=0, bias=False)
        )

        self.shortcut = nn.Sequential()
        if(in_channels != out_channels):
          # Use 1x1 to change the size of the channel for shortcut. This will be summed up with output from other conv layers
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False)
          )

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x1 + x2)
        x = self.conv3(x1 + x2 + x3)
        return x


class TransitionBlock(nn.Module):

    def __init__(self, channels, drop_out):
        super(TransitionBlock, self).__init__()

        self.transition = nn.Sequential(

            # Adding the relu, BN and DO here as the last layer in the previous ConvBlock does not have these, to enable reusability.
            nn.ReLU(),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(drop_out),
            nn.MaxPool2d(2, 2)
        )
        self.channels = channels

    def forward(self, x):
        x = self.transition(x)
        return x


class Quiz9_CIFAR10(nn.Module):

    def __init__(self, channels1=32, channels2=64, channels3=128, drop_out=0.05):
        super(Quiz9_CIFAR10, self).__init__()

        self.input_layer = InputBlock(3, channels1, drop_out)
        self.transition1 = TransitionBlock(channels1, drop_out)
        self.convBlock1 = ConvBlock(channels1, channels2, drop_out)
        self.transition2 = TransitionBlock(channels2, drop_out)
        self.convBlock2 = ConvBlock(channels2, channels3, drop_out)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels3, 10, bias=False)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transition1(x)
        x = self.convBlock1(x)
        x = self.transition2(x)
        x = self.convBlock2(x)
        x = self.gap1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def summarize(self, input):
        summary(self, input_size=input)
