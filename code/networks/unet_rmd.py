import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .BaseModelClass import BaseModel
from torch.autograd import Variable


# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.dropout_p=dropout_p

    def forward(self, x, dropoutflag=False):
        x = self.conv(x)
        if self.training and dropoutflag:
            x=F.dropout(x,self.dropout_p)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch,dropout_p):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
        self.dropout_p = dropout_p

    def forward(self, x, dropoutflag=False):
        x = self.mpconv(x)
        if self.training and dropoutflag:
            x=F.dropout(x,self.dropout_p)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet_rmd(BaseModel):
    def __init__(self, in_channels, classes):
        super(Unet_rmd, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  classes
        self.dropout = [0.05, 0.1, 0.2, 0.3, 0.5]
        #self.dropout = [0.1, 0.2, 0.3, 0.5, 0.5]


        self.inc = InConv(in_channels, 64, self.dropout[0])
        self.down1 = Down(64, 128, self.dropout[1])
        self.down2 = Down(128, 256, self.dropout[2])
        self.down3 = Down(256, 512, self.dropout[3])
        self.down4 = Down(512, 512, self.dropout[4])
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, classes)
        self.outc1 = OutConv(64, classes)
        self.outc2 = OutConv(128, classes)
        self.outc3 = OutConv(256, classes)


    def forward(self, x, mode="weak"):
        shape = x.shape[2:]
        if mode == "strong":
            x1 = self.inc(x, dropoutflag=True)
            x2 = self.down1(x1, dropoutflag=True)
            x3 = self.down2(x2, dropoutflag=True)
            x4 = self.down3(x3, dropoutflag=True)
            x5 = self.down4(x4, dropoutflag=True)
        elif mode == "weak":
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        x = self.up1(x5, x4)
        out3 = self.outc3(x)
        out3 = torch.nn.functional.interpolate(out3, shape)

        x = self.up2(x, x3)
        out2 = self.outc2(x)
        out2 = torch.nn.functional.interpolate(out2, shape)

        x = self.up3(x, x2)
        out1 = self.outc1(x)
        out1 = torch.nn.functional.interpolate(out1, shape)

        x = self.up4(x, x1)
        out = self.outc(x)
        return out ,out1,out2,out3