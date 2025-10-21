# Note: This is the file responsible for actually training the CNN (Convolutional Neural Network) model
# To actually run this file to make the model, run: python expressionClassifier.py (while you're in the venv)

# Imports the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

# Sets up the device (use GPU, and CPU if no GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths for all my dataset directories
dataDir = 'dataset'
trainDir = os.path.join(dataDir, 'train')
valDir = os.path.join(dataDir, 'validation')

# Resizes the images and just normalizes them
trainTransform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # 10 degrees rotation
    transforms.ToTensor(), # Converting image to tensor (the grid thingy)
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Same thing as trainTransform but for validation (and not augmented cuz its for validation)
valTransform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Simple CNN for emotion recognition (increased depth)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # CNN Layers
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3) # Accounts for overfitting
        self.fc1 = nn.Linear(256 * 3 * 3, 512)  # Adjusted the input size for the fully connected layer
        self.fc2 = nn.Linear(512, 7)

# Function for the order that the layers are applied
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))  # Pass through the new conv layer
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    # Load datasets with data augmentation for training
    trainData = datasets.ImageFolder(trainDir, transform=trainTransform)
    valData = datasets.ImageFolder(valDir, transform=valTransform)

    trainLoader = DataLoader(trainData, batch_size=64, shuffle=True, num_workers=4)
    valLoader = DataLoader(valData, batch_size=64, shuffle=False, num_workers=4)

    # Initialize model, loss, optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce learning rate by half every 5 epochs

    # Training loop
    numEpochs = 20  # Increased number of epochs
    for epoch in range(numEpochs):
        model.train()
        totalLoss = 0
        for images, labels in trainLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()

        print(f"Epoch [{epoch+1}/{numEpochs}], Loss: {totalLoss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valLoader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        # Step the learning rate scheduler
        scheduler.step()

    # Save the model (the model is called emotionModel.pth)
    torch.save(model.state_dict(), "emotionModel.pth")
    print("Emotion model saved!")

if __name__ == '__main__':
    main()
