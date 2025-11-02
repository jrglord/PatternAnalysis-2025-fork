# predict.py
import torch
from modules import *
from dataset import *
import torchvision.transforms as transforms
import PIL.Image as Image

# Input image
img_path = "recognition/Alzheimers_Classifier_Jack_Lord/ADNI/AD_NC/test/AD/388206_78.jpeg"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

# Hyper-parameters
num_start_channels = 3
num_classes = 2
width = 256
height = 240

# Local model
model = ConvnextNetwork(num_start_channels, num_classes, width, height, device)

# Load the saved parameters for the model
model.load_state_dict(torch.load('recognition/Alzheimers_Classifier_Jack_Lord/saved_models/saved_model.pth', map_location=device))
model.eval()
model = model.to(device)

# Apply tensor transform to the image and add batch dimension (B in [B, C, H, W]) so that it fits in the model
transform_predict = transforms.Compose([transforms.ToTensor()])
img = Image.open(img_path).convert('RGB')
img = transform_predict(img).unsqueeze(0).to(device)

# Model output
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output.data, 1)

# Print statement
predicted_class = ['AD', 'NC']
print(f"Predicted class for {img_path}: {predicted_class[predicted]}")