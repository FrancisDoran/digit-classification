import torch
import streamlit as st # type: ignore
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class SAFPooling(nn.Module):
    """
    SAF-Pooling: A pooling mechanism that pools the highest activations 
    and suppresses some randomly to improve robustness.
    """
    def __init__(self, pool_size):
        super(SAFPooling, self).__init__()
        self.pool_size = pool_size
    
    def forward(self, x):
        # Max pooling for highest activations
        x_max = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        # Random dropout of some activations
        mask = torch.bernoulli(torch.full_like(x_max, 0.9))  # Keep 90% activations
        return x_max * mask

class ConvBlock(nn.Module):
    """
    A convolutional block with Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(SimpNet, self).__init__()
        self.features = nn.Sequential(
            # Group 1
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            SAFPooling(pool_size=2),  # Output: 64x14x14

            # Group 2
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            SAFPooling(pool_size=2),  # Output: 128x7x7

            # Group 3
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            SAFPooling(pool_size=2)   # Output: 256x3x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),  # Adjusted input size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpNet(num_classes=10, in_channels=1).to(device)
model.load_state_dict(torch.load("best_model-test.pth",weights_only="True"))
model.eval()

# Define the class names (digits 0-9)
classes = [str(i) for i in range(10)]

# Preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to grayscale and resize to 28x28
    image = image.convert('L').resize((28, 28))
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize the image to [0, 1]
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    image = image.to(device)  # Move to the same device as the model
    return image

# Streamlit UI
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess and make a prediction
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(1).item()  # Get the predicted class
        st.write(f"Prediction: {classes[prediction]}")

