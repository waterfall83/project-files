import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
#import gmi
#from medmnist import ChestMNIST
import numpy as np
import os

#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load preprocessed noisy training data from .pt file
NOISY_DATA_PATH = os.path.join("noisy_data","train_data.pt")
data = torch.load(NOISY_DATA_PATH)
images = data['images'] #shape: [N, 1, 28, 28]
labels = data['labels'].float() #[N, 14]

#creating dataset and dataloader
noisy_dataset = TensorDataset(images, labels)
trainloader = DataLoader(noisy_dataset, batch_size=32, shuffle=True, num_workers=0)

#cnn model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  #1 input channel - grayscale
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

#loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#training loop
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #labels = labels.squeeze()  #making sure labels are 1D

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished training!")

#save trained model weights
torch.save(net.state_dict(), "noisy_classifier_weights.pth")
print("Saved model weights to 'noisy_classifier_weights.pth'")
