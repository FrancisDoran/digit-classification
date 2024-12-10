import torch
import streamlit as st # type: ignore
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
import numpy as np

# Define the model (same as in the original code)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("semeion.data",weights_only="True"))
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

