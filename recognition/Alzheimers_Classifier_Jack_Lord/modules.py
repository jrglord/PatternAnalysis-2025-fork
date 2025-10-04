# modules.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

class ConvnextBlock(nn.Module):
    def __init__(self, ch_dw, kernel_size, stride=1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, dropout_p=0.3):
        super().__init__()

        # Structure:

        # Depthwise Conv2d
        self.depthwiseConv = nn.Conv2d(in_channels=ch_dw, out_channels=ch_dw, kernel_size=7, stride=1)

        # Layer Norm
        self.layer_norm = 

        # Conv2d
        self.conv_1 = 

        # GELU
        self.act_layer = act_layer()

        # Conv2d
        self.conv_2 = 

        # Layer Scale
        self.layer_scale = 

        # Drop Path
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.layer_norm(self.depthwiseConv(x))
        out = self.conv_1(out)
        out = self.act_layer(out)
        out = self.conv_2(out)
        out = self.layer_scale(out)
        out = self.dropout(out)
        return out
