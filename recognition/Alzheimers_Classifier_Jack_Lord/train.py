# train.py
import torch
from modules import *
from dataset import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# Hyper-parameters
channels = 3
num_classes = 2
width = 256
height = 240


model = ConvnextNetwork(channels, num_classes, width, height)
