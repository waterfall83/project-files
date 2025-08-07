import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage, Compose, ToTensor
from medmnist import ChestMNIST
import numpy as np

# Load denoised data
denoised_data = torch.load("denoised_data.pt")['images']

# Load noisy/original dataset
dataset = ChestMNIST(split='train', download=True, transform=ToTensor())

to_pil = ToPILImage()

plt.figure(figsize=(10, 4))
for i in range(5):
    noisy_tensor = dataset[i][0]  # Already a tensor
    denoised_tensor = denoised_data[i]  # Tensor

    # Plot noisy
    plt.subplot(2, 5, i + 1)
    plt.imshow(to_pil(noisy_tensor), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Plot denoised
    plt.subplot(2, 5, i + 6)
    plt.imshow(to_pil(denoised_tensor), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()
