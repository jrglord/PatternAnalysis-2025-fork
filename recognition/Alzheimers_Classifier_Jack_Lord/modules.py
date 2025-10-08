# modules.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import random

class ConvnextBlock(nn.Module):
    def __init__(self, ch_dw, device, act_layer=nn.GELU, layer_scale_init_val=1e-6, drop_path_prob = 0.1):
        super().__init__()

        random.seed(42)
        self.device = device

        # Structure:

        # Depthwise Conv2d
        self.depthwiseConv = nn.Conv2d(in_channels=ch_dw, out_channels=ch_dw, kernel_size=7, stride=1, groups=ch_dw, padding=3)

        # Layer Norm

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

        # Layer Norm
        layer_norm = nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]]).to(self.device)

        out = layer_norm(self.depthwiseConv(x))
        out = self.conv_1(out)
        out = self.act_layer(out)
        out = self.conv_2(out)
        out = self.layer_scale.view(1, -1, 1, 1) * out
        out = self.drop_path_val * out
        out = x + out
        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, in_ch, device):
        super().__init__()
        self.device = device
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2, kernel_size=2, stride=2)

    def forward(self, x):
        layer_norm = nn.LayerNorm([x.shape[1], x.shape[2], x.shape[3]]).to(self.device)
        out = layer_norm(x)
        out = self.conv(out)
        return out
    
class ConvnextNetwork(nn.Module):
    def __init__(self, in_ch, num_classes, width, height, device):
        super().__init__()
        
        self.in_channels = in_ch
        self.after_stem_num_chs = in_ch*32
        self.after_stem_width = int(width/4)
        self.after_stem_height = int(height/4)
        self.layer_before_pool_width = int(width/32)
        self.layer_before_pool_height = int(height/32)
        self.final_num_chs = self.after_stem_num_chs*8
        self.device = device

        self.init_conv = nn.Conv2d(in_channels=in_ch, out_channels=self.after_stem_num_chs, kernel_size=4, stride=4)
        self.glob_avg_pool = nn.AvgPool2d((self.layer_before_pool_height, self.layer_before_pool_width))
        self.linear_layer = nn.Linear(self.final_num_chs, num_classes)
        self.softmax_layer = nn.Softmax()

    def forward(self, x):
        print(x.shape)
        out = self.init_conv(x)
        print(out.shape)
        layer_norm = nn.LayerNorm([self.after_stem_num_chs, self.after_stem_height, self.after_stem_width]).to(self.device)
        out = layer_norm(out)
        
        current_channels = self.after_stem_num_chs
        conv_block = ConvnextBlock(current_channels, self.device)
        ds_block = DownSamplingBlock(current_channels, self.device)

        for i in range(3):
            out = conv_block(out)
        out = ds_block(out)

        current_channels = current_channels*2
        conv_block = ConvnextBlock(current_channels, self.device)
        ds_block = DownSamplingBlock(current_channels, self.device)

        for i in range(3):
            out = conv_block(out)
        out = ds_block(out)

        current_channels = current_channels*2
        conv_block = ConvnextBlock(current_channels, self.device)
        ds_block = DownSamplingBlock(current_channels, self.device)

        for i in range(9):
            out = conv_block(out)
        out = ds_block(out)

        current_channels = current_channels*2
        conv_block = ConvnextBlock(current_channels, self.device)
        ds_block = DownSamplingBlock(current_channels, self.device)

        for i in range(3):
            out = conv_block(out)
        out = ds_block(out)

        out = self.glob_avg_pool(out)
        layer_norm = nn.LayerNorm([self.final_num_chs, 1, 1]).to(self.device)
        out = layer_norm(out)
        out = self.linear_layer(out)
        out = self.softmax_layer(out)
        return out