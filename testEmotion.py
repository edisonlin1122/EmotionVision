# This file is used to test the model (emotionModel.pth) on an image inputted by the user (outside of the training and validation datasets)

# Import necessary modules
import torch
from torchvision import transforms
from PIL import Image
from expressionClassifier import EmotionCNN  # Import the model from your training script
import tkinter as tk
from tkinter import filedialog

# Device setup (GPU and CPU if no GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loads the model (emotionModel.pth)
model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotionModel.pth"))
model.eval()  # sets model to eval mode 'cause we're testing

# The transformations for the input image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function for the emotion prediction
def predict_emotion(image_path):
    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Run img through model to get its prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Maps index to emotion label
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    predicted_emotion = emotion_labels[predicted.item()]

    return predicted_emotion

# Open file explorer for user to select an image
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

# Ensures this only works if an image input was used
if image_path:
    predicted_emotion = predict_emotion(image_path)
    print(f"Predicted Emotion: {predicted_emotion}")
else:
    print("No image selected!")
