# modules.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

class ConvnextBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, dropout_p=0.3):
        # Structure:
        # Depthwise Conv2d
        # Layer Norm
        # Conv2d
        # GELU
        # Conv2d
        # Layer Scale
        # Drop Path

        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.norm1(self.features(x))
        out = self.act_layer(out)
        out = self.dropout(out)
        return out
