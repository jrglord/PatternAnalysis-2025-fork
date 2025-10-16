# train.py
import torch
from modules import *
from dataset import *
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt

# Setting up pyplot
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, 1)  # x-axis range
ax.set_ylim(0, 1)  # y-axis range
x_data = [] # number of iterations
y_data = [] # loss in each iteration

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# Hyper-parameters
num_epochs = 3
learning_rate = 4e-3
num_start_channels = 3
num_classes = 2
width = 256
height = 240

# Local model
model = ConvnextNetwork(num_start_channels, num_classes, width, height, device)

# Load pre-trained model and weights
# weights = ConvNeXt_Tiny_Weights.DEFAULT
#IMAGENET1K_V1
# model = convnext_tiny(weights)
# model.classifier[2] = nn.Linear(768, 2)

# Freezes all weights except classification layer
# for param in model.features.parameters():
#     param.requires_grad = False



model = model.to(device)

criterion = nn.CrossEntropyLoss()
total_step = len(train_loader)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)

model.train()
print("> Training")
start = time.time() #time generation
max_loss = 0
for epoch in range(num_epochs):
    epoch_loss_sum = 0
    for i, (images, labels) in enumerate(train_loader):
        ax.set_xlim(0, i+1)  # x-axis range
        ax.set_ylim(0, max_loss+0.1)  # y-axis range
        
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item()

        print(f"i: {i}, loss: {loss.item()}")

        # Update max loss and iteration plot
        x_data.append(i+epoch*36)
        y_data.append(loss.item())
        
        if loss.item() > max_loss:
            max_loss = loss.item()
        

        ax.clear()  # Clear previous frame
        ax.plot(x_data, y_data, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Convergence Plot")

    print ("Epoch [{}/{}], Avg. Loss: {:.5f}".format(epoch+1, num_epochs, epoch_loss_sum/(i+1)))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
plt.ioff()  # Turn off interactive mode


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
plt.show()

"""
25% of dataset:

Training took 5748.528654336929 secs or 95.80881090561549 mins in total
> Testing
Test Accuracy: 47.68888888888889 %
Testing took 701.1007871627808 secs or 11.68501311937968 mins in total
"""

"""
25% of dataset (no softmax):

Training took 5493.959788799286 secs or 91.5659964799881 mins in total
> Testing
Test Accuracy: 48.4 %
Testing took 96.76594185829163 secs or 1.6127656976381937 mins in total
"""

"""
Training did better when only accessing the final layer vs the whole set of weights
"""

# If loss is plateauing but chaotic then add dropout layers
# If loss is plateauing and smooth then increase learning rate (maybe add a scheduler ot the lr) 0.5 is big lr wind down to 0.003
# model.save and model.load
# include saved model files in gitignore
