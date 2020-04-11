from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvMaxPoolBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvMaxPoolBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class InputLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InputLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class S11Resnet(nn.Module):

    def __init__(self):
        super(S11Resnet, self).__init__()

        self.input_layer = InputLayer(3, 64)
        self.layer1_conv_maxpool = ConvMaxPoolBlock(64, 128)
        self.layer1_resnet_block = ResnetBlock(128, 128)
        self.layer2 = ConvMaxPoolBlock(128, 256)
        self.layer3_conv_maxpool = ConvMaxPoolBlock(256, 512)
        self.layer3_resnet_block = ResnetBlock(512, 512)

        self.max_pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.input_layer(x)
        # Layer 1
        x = self.layer1_conv_maxpool(x)
        r = self.layer1_resnet_block(x)
        x = x + r

        # Layer 2
        x = self.layer2(x)

        # Layer 3
        x = self.layer3_conv_maxpool(x)
        r = self.layer3_resnet_block(x)
        x = x + r

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)
