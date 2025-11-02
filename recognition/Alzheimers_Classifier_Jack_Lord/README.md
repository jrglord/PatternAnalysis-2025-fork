# Model Architecture and Problem
ConvNeXT is a modern Convolutional Neural Network (CNN) architecture that enhances traditional CNNs by integrating the design philosophy of Vision Transformers (ViTs) [1]. Vision Transformers are capable of achieving image recognition and classification by processing images as a series of patches and subsequently uses self-attention layers in order to learn the relationships between different pairs of patches across the image regardless of their proximity to each other. This provides the unique advantage of allowing training on broader patterns across the whole image, improving flexibility and accuracy, while traditional CNNs are limited to training based on patterns within close proximity to elements. ConvNeXT seeks to combine these two approaches by replicating this self-attention mechanism with the structure of typical CNNs. It achieves this by patchifying the image into 4x4 patches by beginning with a 4x4 convolution with stride 4, preventing overlap. Another modification inspired by Vision Transformer design was the replacement of ReLU (Rectified Linear Unit) as an activation layer (typically used in CNNs) with GELU (Gaussian Error Linear Unit) due to its implementation in ViTs [2].

An example of a typical ConvNeXT structure is shown in the figure below, although for the purposes of this project's specific task, image input dimensions and subsequent block dimensions were modified, while the overall structure remained intact. The ConvNeXT block architecture and downsampling block architecture are displayed in figures 2 and 3.

![ConvNeXt-structure](https://github.com/user-attachments/assets/2107b58f-5e82-47c4-b4b0-34dcaa33c4dc)
Figure 1: ConvNeXT network architecture. Source: [1]
<br>
<br>
<br>
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/95c679b1-ee69-480e-abe1-da7ee014ca0c" />
Figure 2: ConvNeXT block architecture. Source: [1]
<br>
<br>
<br>
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/46d5e094-5982-4811-bce7-6bbb93dad9dd" />
Figure 3: Downsampling block architecture. Source: [1]
<br>
<br>
<br>


# Dependencies
In order to run this program the following dependencies need to be installed. Commands for installation are prescribed below:
## matplotlib 3.10.7
```
pip install matplotlib
```
## Pillow 12.0.0
```
pip install Pillow
```
## scikit_learn 1.7.2
```
pip install scikit_learn
```
## torch 2.8.0+cu126 and torchvision 0.23.0+cu126
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

# How to Run
## train.py
This program trains the network and subsequently tests it against a testing dataset.
```
python train.py
```
## predict.py
This program provides the trained model's output for a single image file as input
```
python predict.py
```
# Pre-proccessing and Transforms
When loading the images into datasets, image transforms are applied during preprocessing to convert the images into tensors and add variation to ensure generalisability. For both testing and training datasets, `transforms.ToTensor()` is applied. For the training dataset, transforms of `transforms.RandomHorizontalFlip(p=0.3)` and `transforms.RandomRotation(15)` are also applied to ensure that 30% of the training images are flipped and that the images are rotated randomly within the range of -15 degrees to 15 degrees, to improve generalisability.

# Training Settings
## Class weights
Weights were applied to each class to account for the differences in number of images between classes in the training set (class AD had 10,400 images, while class NC had 11,120 images). This ensured that losses in each class had the same impact on the loss function.
## Loss Function

## Optimiser
## Scheduler

# Example run
## train.py
## predict.py

# Bibliography
[1] 	GeeksforGeeks, “ConvNeXt,” SanchhayaEducation Private Limited, 15 July 2025. [Online]. Available: https://www.geeksforgeeks.org/computer-vision/convnext/. [Accessed 2 November 2025].

[2] 	Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell and S. Xie, “A ConvNet for the 2020s,” 2 Mar 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2201.03545.


