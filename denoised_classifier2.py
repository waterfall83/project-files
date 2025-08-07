import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import gmi
from medmnist import PathMNIST
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained denoiser model
denoiser = gmi.network.SimpleCNN(
    input_channels=3,
    output_channels=3,
    hidden_channels_list=[16, 32, 64, 128, 256, 128, 64, 32, 16],
    activation=torch.nn.SiLU(),
    dim=2
).to(device)

print("Loaded denoiser")

# Load weights
denoiser.load_state_dict(torch.load('denoiser.pth'))
denoiser.eval()

# Load clean dataset
transform = transforms.ToTensor()
clean_dataset = PathMNIST(split="train", download=True, transform=transform)

# Denoise only first 2000 images
denoised_imgs = []
labels = []

print("About to denoise 2000 images...")

with torch.no_grad():
    for idx, (img, lbl) in enumerate(tqdm(clean_dataset, total=2000, desc="Denoising images", ncols=100)):
        if idx >= 2000:
            break
        img = img.unsqueeze(0).to(device)
        denoised = denoiser(img).squeeze(0).detach().cpu().numpy()
        denoised_imgs.append(denoised)
        labels.append(lbl)

# Convert to tensors
denoised_imgs = torch.tensor(np.array(denoised_imgs))  # Shape: [2000, 3, 28, 28]
labels = torch.tensor(np.array(labels)).long().squeeze()  # Shape: [2000]

print("Created dataset")

# Create dataset and dataloader
denoised_dataset = TensorDataset(denoised_imgs, labels)
trainloader = DataLoader(denoised_dataset, batch_size=32, shuffle=True)

# Define classifier
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train classifier
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Starting training...")
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.squeeze())  # Squeeze labels to ensure 1D
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished training!")

# Save model
torch.save(net.state_dict(), 'denoised_classifier_weights.pth')
print("Model weights saved to 'denoised_classifier_weights.pth'")

# Evaluate on test set
testset = PathMNIST(split="test", download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

correct = 0
total = 0
net.eval()

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.squeeze()  # ensure shape is [batch_size]
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total
print(f"Accuracy on clean test set: {accuracy:.2f}%")
