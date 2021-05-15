import torch.nn as nn
import torch

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def init_layers(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class UNet(nn.Module):
    def __init__(self, dropout=0.2):
        super(UNet, self).__init__()

        self.start = double_conv(1, 32) 
        self.down1 = double_conv(32, 64) 
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 128)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout  = nn.Dropout2d(p=dropout)

        self.out = nn.Sequential(
            nn.Linear(32768, 8)
        )
        
        for el in [self.start, self.down1, self.down2, self.down3]:
            el.apply(init_layers)
        
        self.out.apply(init_layers)

    def forward(self, input):
        conv1 = self.start(input) # 1x128x128 -> 16x128x128
        x = self.max_pool(conv1)  # 16x64x64
#         x = self.dropout(x)
        
        conv2 = self.down1(x)     # 16x64x64 -> 32x64x64
        x = self.max_pool(conv2)  # 32x32x32
#         x = self.dropout(x)

        conv3 = self.down2(x)     # 32x32x32 -> 64x32x32
        x = self.max_pool(conv3)  # 64x16x16
#         x = self.dropout(x)
        
        x = self.down3(x)         # 64x16x16 -> 64x16x16
#         x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        
        return self.out(x)
    