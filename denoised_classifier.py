# CLASSIFIER TRAINED ON DENOISED IMAGES

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from medmnist import ChestMNIST
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-denoised images and labels
print("Loading denoised data...")
data = torch.load('denoised_data.pt')
denoised_imgs = data['images']
labels = data['labels']

#labels = labels.long()

# DEBUG: print original labels shape and dtype
print(f"Original labels shape: {labels.shape}, dtype: {labels.dtype}")

# If labels are one-hot encoded (shape [N, 14]), convert to class indices [N]
if labels.ndim > 1 and labels.shape[1] > 1:
    labels = labels.argmax(dim=1)
    print("Converted one-hot labels to class indices.")

# Convert labels to LongTensor as required by nn.CrossEntropyLoss
labels = labels.long()
print(f"Processed labels shape: {labels.shape}, dtype: {labels.dtype}")


# Create dataset and dataloader
denoised_dataset = TensorDataset(denoised_imgs, labels)
trainloader = DataLoader(denoised_dataset, batch_size=32, shuffle=True)

print("Denoised data loaded and batched.")

# Define classifier
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)  #ChestMNIST has 14 labels

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
print("Starting training...")
for epoch in range(3):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = criterion(outputs, labels.squeeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.3f}")
            running_loss = 0.0

print("Finished training.")

# Save model
torch.save(net.state_dict(), "denoised_classifier_weights.pth")
print("Model saved to 'denoised_classifier_weights.pth'.")

# Evaluation
transform = transforms.ToTensor()
testset = ChestMNIST(split="test", download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

net.eval()
# Testing
correct = 0
total = 0

net.eval()  # Make sure the model is in eval mode
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.argmax(dim=1).to(device) #new

        # Ensure labels are 1D
        if labels.ndim == 2:
            labels = labels.squeeze(1)

        outputs = net(images)
        #_, predicted = torch.max(outputs, 1)

        '''total += labels.size(0)
        correct += (predicted == labels).sum().item()'''
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
print(f'Correct: {correct}, Total: {total}')
