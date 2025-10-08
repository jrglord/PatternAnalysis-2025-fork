import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

#Data

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/train', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/test', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
