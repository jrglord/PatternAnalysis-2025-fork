# train.py
import torch
from modules import *
from dataset import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# Hyper-parameters
num_epochs = 5
learning_rate = 1e-3
num_start_channels = 3
num_classes = 2
width = 256
height = 240

model = ConvnextNetwork(num_start_channels, num_classes, width, height, device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        print("i: ",i)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")


"""
25% of dataset:

Training took 5748.528654336929 secs or 95.80881090561549 mins in total
> Testing
Test Accuracy: 47.68888888888889 %
Testing took 701.1007871627808 secs or 11.68501311937968 mins in total
"""