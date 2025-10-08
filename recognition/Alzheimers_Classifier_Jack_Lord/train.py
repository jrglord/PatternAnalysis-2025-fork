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
num_start_channels = 1
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
        images = images.to(device)
        labels = labels.to(device)
        print("images:")
        print(images.shape)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")