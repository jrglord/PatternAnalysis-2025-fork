import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

torch.manual_seed(42)

#Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

full_trainset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/train', transform=transform)

train_subset_size = len(full_trainset)//4
train_subset_indices = torch.randperm(len(full_trainset))[:train_subset_size]
subset_trainset = torch.utils.data.Subset(full_trainset, train_subset_indices)

train_loader = torch.utils.data.DataLoader(subset_trainset, batch_size=200, shuffle=True)


full_testset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/test', transform=transform)

test_subset_size = len(full_testset)//4
test_subset_indices = torch.randperm(len(full_testset))[:test_subset_size]
subset_testset = torch.utils.data.Subset(full_testset, test_subset_indices)

test_loader = torch.utils.data.DataLoader(subset_testset, batch_size=200, shuffle=False)
