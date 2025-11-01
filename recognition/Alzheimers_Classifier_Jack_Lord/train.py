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

# Load pre-trained model and weights
# weights = ConvNeXt_Tiny_Weights.DEFAULT
#IMAGENET1K_V1
# model = convnext_tiny(weights)
# model.classifier[2] = nn.Linear(768, 2)

# Freezes all weights except classification layer
# for param in model.features.parameters():
#     param.requires_grad = False



model = model.to(device)

counts = Counter([label for _, label in full_trainset.samples])

class_num_samples = torch.tensor([counts[0], counts[1]])
class_weights = 1. / class_num_samples.float()
class_weights = class_weights / class_weights.sum()

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
total_step = len(train_loader)
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
scheduler = StepLR(optimiser, step_size=1, gamma=0.9)
print(full_trainset.class_to_idx)
model.train()
print("> Training")
start = time.time() #time generation
max_loss = 0
total_iter = 0
for epoch in range(num_epochs):
    epoch_loss_sum = 0
    print(f"Epoch {epoch+1} start:")
    for i, (images, labels) in enumerate(train_loader):
        total_iter += 1
        ax.set_xlim(0, i+1)  # x-axis range
        ax.set_ylim(0, max_loss+0.1)  # y-axis range
        
        images = images.to(device)
        labels = labels.to(device)

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

cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AD', 'NC'])
disp.plot(cmap=plt.cm.Blues)
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
