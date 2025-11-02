# train.py
import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from modules import *
from dataset import *
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
num_epochs = 10
learning_rate = 1e-4
num_start_channels = 3
num_classes = 2
width = 256
height = 240

# Local model
model = ConvnextNetwork(num_start_channels, num_classes, width, height, device)
model = model.to(device)

# Counting the number of images in each class and then calculating weights for each class based on image sample sizes
counts = Counter([label for _, label in full_trainset.samples])
class_num_samples = torch.tensor([counts[0], counts[1]])
class_weights = 1. / class_num_samples.float()
class_weights = class_weights / class_weights.sum()

# Loss function, optimiser, and scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
#total_step = len(train_loader)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
scheduler = StepLR(optimiser, step_size=1, gamma=0.9)

model.train()
print("> Training")

start = time.time() #time generation
max_loss = 0
total_iter = 0

# Loop through each epoch
for epoch in range(num_epochs):
    epoch_loss_sum = 0
    print(f"Epoch {epoch+1} start:")

    # Loop through each batch
    for i, (images, labels) in enumerate(train_loader):
        total_iter += 1
        ax.set_xlim(0, i+1)  # x-axis range
        ax.set_ylim(0, max_loss+0.1)  # y-axis range
        
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.long()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        epoch_loss_sum += loss.item()

        print(f"batch: {i+1}, lr: {scheduler.get_lr()}, loss: {loss.item()}")

        # Update max loss and iteration plot
        x_data.append(total_iter)
        y_data.append(loss.item())
        
        if loss.item() > max_loss:
            max_loss = loss.item()
        

        ax.clear()  # Clear previous frame
        ax.plot(x_data, y_data, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Convergence Plot")
        
    scheduler.step()
    
    print ("Epoch [{}/{}], Avg. Loss: {:.5f}".format(epoch+1, num_epochs, epoch_loss_sum/(i+1)))

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
plt.ioff()  # Turn off interactive mode

# Save the model parameters
torch.save(model.state_dict(), 'recognition/Alzheimers_Classifier_Jack_Lord/saved_models/saved_model.pth')
print("> Model saved")

# Test the model
print("> Testing")
all_predictions = [] #predicted for confusion matrix
all_labels = [] #labels for confusion matrix
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

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print('Test Accuracy: {} %'.format(100 * correct / total))

end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
plt.show()

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AD', 'NC'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

