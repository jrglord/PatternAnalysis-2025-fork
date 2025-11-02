#dataset.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import ConcatDataset

torch.manual_seed(42)

# Transforms for training dataset
transform = transforms.Compose([transforms.ToTensor(),  transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(15)])

# Training dataset and data loader
full_trainset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/train', transform=transform)

train_subset_size = len(full_trainset)//1
train_subset_indices = torch.randperm(len(full_trainset))[:train_subset_size]
subset_trainset = torch.utils.data.Subset(full_trainset, train_subset_indices)

train_loader = torch.utils.data.DataLoader(subset_trainset, batch_size=32, shuffle=True)

# Testing dataset and data loader
full_testset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/test', transform=transforms.Compose([transforms.ToTensor()]))

test_subset_size = len(full_testset)//1
test_subset_indices = torch.randperm(len(full_testset))[:test_subset_size]
subset_testset = torch.utils.data.Subset(full_testset, test_subset_indices)

test_loader = torch.utils.data.DataLoader(subset_testset, batch_size=32, shuffle=False)
