import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
#import skimage


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
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    auc_per_class = []
    for i in range(all_targets.shape[1]):
        try:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
        except ValueError:
            auc = float('nan')  # if class has no positive samples in test set
        auc_per_class.append(auc)

    print("AUC scores per class:")
    print(auc_per_class)
    print("Finished Testing")

    return auc_per_class

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load denoised test set from saved file instead of raw ChestMNIST test
    '''denoised_images, denoised_labels = torch.load("chestmnist_test_denoised.pt")
    denoised_dataset = TensorDataset(denoised_images, denoised_labels)
    test_loader = DataLoader(denoised_dataset, batch_size=32, shuffle=False)'''

    data = torch.load("noisy_data/test_data.pt")
    test_images = data['images']
    test_labels = data['labels'].float()
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Net()
    model.load_state_dict(torch.load('noisy_classifier_weights.pth', map_location=device))

    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
