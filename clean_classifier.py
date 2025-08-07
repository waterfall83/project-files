
import torch
import torchvision.transforms as transforms
from medmnist import ChestMNIST
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#def train():
transform = transforms.Compose([
transforms.ToTensor(),
])
dataset = ChestMNIST(split="train", download=True, size=28, transform=transform)
#convert_to_tensor = transforms.ToTensor()
#tensor_image = convert_to_tensor(dataset[0][0])
#print(tensor_image.shape)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=32,shuffle=True, num_workers=0)

_, label = dataset[0]
print("Sample label:", label)
print("Shape of label:", label.shape)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.to(torch.float)
        inputs, labels = inputs.to(device), labels.to(device)

        # Print inputs and labels shapes to debug
        #print(f"Batch {i}: Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

    # Skip empty batches if they exist
        if inputs.size(0) == 0 or labels.size(0) == 0:
            print(f"Skipping batch {i} because inputs or labels are empty.")
            continue
    
        if inputs.size(0) != labels.size(0):
            print(f"Skipping batch {i} due to size mismatch. Inputs: {inputs.size(0)}, Labels: {labels.size(0)}")
            continue
        
        #labels = labels.squeeze()
        #print(f"Labels after squeeze: {labels.shape}")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(f"Outputs shape: {outputs.shape}")

        #print("random for loop")
        if (i < 1):
            #print(outputs.shape)
            #print(labels.shape)
            print("hi")
            #print(trainloader.shape())
        loss = criterion(outputs, labels)
        #line 83 has the error

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
            
        #if i % 2000 == 1999:    # print every 2000 mini-batches
        if i % 500 == 499:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
            running_loss = 0.0

print('Finished Training')

print("Finished training.")
torch.save(net.state_dict(), 'clean_classifier_weights.pth')
print("Model weights saved to 'clean_classifier_weights.pth'.")

#testing
correct = 0
total = 0

dataset_test = ChestMNIST(split="test", download=True, size=28, transform=transform)
testloader = torch.utils.data.DataLoader(dataset_test, batch_size=32,shuffle=True, num_workers=0)

net.eval()

'''with torch.no_grad():
    #for data in testloader:
    for images, labels in testloader:
        #images, labels = data
        outputs = net(images)
        labels = labels.to(torch.float)
        labels = labels.to(device)  # Shape: [batch_size, 14]
        
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        #_, predicted = torch.max(outputs,1)

        #labels = labels.squeeze(1)
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()


    print(f'Accuracy of the network on the 1000 test images: {100 * correct // total} %')
    print(f'total is {total} and correct is {correct}')'''

all_probs = [] #predictions of network
all_targets = [] #actual labels

#targets is the same as labels
for images, labels in testloader:
    #move images and labels to device
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(net(images))
    all_probs.append(probs.cpu())
    all_targets.append(labels.cpu().float())

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()
#shape should be num_samples, num_labels
from sklearn.metrics import roc_auc_score

# Multi-label AUC: compute per class
auc_per_class = []
for i in range(all_targets.shape[1]):
    auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
    auc_per_class.append(auc)

print(auc_per_class)

print('Finished Testing')

#if __name__ == "__main__":
    #train()

print()
