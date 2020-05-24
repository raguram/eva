import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down_sample(out), out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class ResUNet_Lite(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super(ResUNet_Lite, self).__init__()

        self.enc_res1 = ResBlock(in_channels, 64)
        self.enc_res2 = ResBlock(64, 128)

        self.double_conv = DoubleConv(128, 256)

        self.dec_up_mask2 = UpBlock(128 + 256, 128)
        self.dec_up_mask1 = UpBlock(64 + 128, 64)
        self.mask_pred = nn.Conv2d(64, out_channels, kernel_size=1)

        self.dec_up_depth2 = UpBlock(128 + 256, 128)
        self.dec_up_depth1 = UpBlock(64 + 128, 64)
        self.depth_pred = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x, skip1_out = self.enc_res1(x)
        x, skip2_out = self.enc_res2(x)
        x = self.double_conv(x)

        # Mask decoder
        m = self.dec_up_mask2(x, skip2_out)
        m = self.dec_up_mask1(m, skip1_out)
        m = self.mask_pred(m)

        # Depth decoder
        d = self.dec_up_depth2(x, skip2_out)
        d = self.dec_up_depth1(d, skip1_out)
        d = self.depth_pred(d)

        return m, d


class ResUNet_Dual(nn.Module):
    """
    Hybrid solution of resnet blocks and double conv blocks
    """

    def __init__(self, in_channels, out_channels=1):
        super(ResUNet_Dual, self).__init__()

        self.enc_res1 = ResBlock(in_channels, 32)
        self.enc_res2 = ResBlock(32, 64)
        self.enc_res3 = ResBlock(64, 128)
        self.enc_res4 = ResBlock(128, 256)

        self.double_conv = DoubleConv(256, 512)

        self.dec_up_mask4 = UpBlock(256 + 512, 256)
        self.dec_up_mask3 = UpBlock(128 + 256, 128)
        self.dec_up_mask2 = UpBlock(64 + 128, 64)
        self.dec_up_mask1 = UpBlock(64 + 32, 32)
        self.mask_pred = nn.Conv2d(32, out_channels, kernel_size=1)

        self.dec_up_depth4 = UpBlock(256 + 512, 256)
        self.dec_up_depth3 = UpBlock(128 + 256, 128)
        self.dec_up_depth2 = UpBlock(64 + 128, 64)
        self.dec_up_depth1 = UpBlock(64 + 32, 32)
        self.depth_pred = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x, skip1_out = self.enc_res1(x)
        x, skip2_out = self.enc_res2(x)
        x, skip3_out = self.enc_res3(x)
        x, skip4_out = self.enc_res4(x)
        x = self.double_conv(x)

        # Mask decoder
        m = self.dec_up_mask4(x, skip4_out)
        m = self.dec_up_mask3(m, skip3_out)
        m = self.dec_up_mask2(m, skip2_out)
        m = self.dec_up_mask1(m, skip1_out)
        m = self.mask_pred(m)

        # Depth decoder
        d = self.dec_up_depth4(x, skip4_out)
        d = self.dec_up_depth3(d, skip3_out)
        d = self.dec_up_depth2(d, skip2_out)
        d = self.dec_up_depth1(d, skip1_out)
        d = self.depth_pred(d)

        return m, d
