import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
  )

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.start = double_conv(3, 112)
        self.down1 = double_conv(112, 224)
        self.down2 = double_conv(224, 448)
        self.down3 = double_conv(448, 448)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout  = nn.Dropout2d(p=0.3)

        self.up3 = double_conv(448 + 448, 224)
        self.up2 = double_conv(224 + 224, 112)
        self.up1 = double_conv(112 + 112, 112)

        self.out = nn.Sequential(
            nn.Conv2d(112, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv1 = self.start(input)
        x = self.max_pool(conv1)

        conv2 = self.down1(x)
        x = self.max_pool(conv2)

        conv3 = self.down2(x)
        x = self.max_pool(conv3)

        x = self.down3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dropout(x)

        x = self.up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dropout(x)

        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dropout(x)

        x = self.up1(x)

        return self.out(x)