# modules.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import random

class ConvnextBlock(nn.Module):
    def __init__(self, ch_dw, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale_init_val=1e-6, drop_path_prob = 0.1):
        super().__init__()

        random.seed(42)

        # Structure:

        # Depthwise Conv2d
        self.depthwiseConv = nn.Conv2d(in_channels=ch_dw, out_channels=ch_dw, kernel_size=7, stride=1, groups=ch_dw)

        # Layer Norm
        self.layer_norm = norm_layer()

        # Conv2d
        self.conv_1 = nn.Conv2d(in_channels=ch_dw, out_channels=4*ch_dw, kernel_size=1, stride=1)

        # GELU
        self.act_layer = act_layer()

        # Conv2d
        self.conv_2 = nn.Conv2d(in_channels=4*ch_dw, out_channels=ch_dw, kernel_size=1, stride=1)

        # Layer Scale
        self.layer_scale = nn.Parameter(torch.ones(ch_dw*layer_scale_init_val))

        # Drop Path
        self.drop_path_val = 1
        if random.random() < drop_path_prob:
            self.drop_path_val = 0

    def forward(self, x):
        out = self.layer_norm(self.depthwiseConv(x))
        out = self.conv_1(out)
        out = self.act_layer(out)
        out = self.conv_2(out)
        out = self.layer_scale.view(1, -1, 1, 1) * out
        out = self.drop_path_val * out
        out = x + out
        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, in_ch, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layer_norm = norm_layer()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.conv(out)
        return out