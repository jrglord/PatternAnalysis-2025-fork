import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

torch.manual_seed(42)

#Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.11559005826711655, 0.2252865880727768)])
# transforms.Resize(256), transforms.CenterCrop(224), 

full_trainset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/train', transform=transform)

# full_trainset_B = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/train', transform = transforms.ToTensor())
# train_loader_full = torch.utils.data.DataLoader(full_trainset_B, batch_size=150, shuffle=False)
# mean = 0
# std = 0
# i=1
# for batch, _ in train_loader_full:
#     print(i)
#     i+=1
#     mean += batch.mean()
#     std += batch.std()
# mean /= len(train_loader_full)
# std /= len(train_loader_full)
# print(f"mean: {mean}, std: {std}")

train_subset_size = len(full_trainset)//4
train_subset_indices = torch.randperm(len(full_trainset))[:train_subset_size]
subset_trainset = torch.utils.data.Subset(full_trainset, train_subset_indices)

train_loader = torch.utils.data.DataLoader(subset_trainset, batch_size=32, shuffle=True)


full_testset = torchvision.datasets.ImageFolder(root='recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/test', transform=transform)

test_subset_size = len(full_testset)//4
test_subset_indices = torch.randperm(len(full_testset))[:test_subset_size]
subset_testset = torch.utils.data.Subset(full_testset, test_subset_indices)

test_loader = torch.utils.data.DataLoader(subset_testset, batch_size=32, shuffle=False)
