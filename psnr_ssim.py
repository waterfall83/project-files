import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import skimage
from medmnist import ChestMNIST
import torchvision.transforms as transforms

transform = transforms.Compose([
transforms.ToTensor(),
])

def evaluate_model(dataloader, noisy_set, denoised_set, device):

    clean_images = []
    labels = []

    with torch.no_grad():
        for images, _labels in dataloader:
            images, _labels = images.to(device), _labels.to(device).float()
            clean_images.append(images)
            labels.append(_labels)

    clean_images = torch.cat(clean_images, dim=0)
    labels = torch.cat(labels, dim=0)
    print("Debug: clean", clean_images.shape)
    print("Debug: noisy", noisy_set.shape)
    print("Debug: denoised", denoised_set.shape)
    print("Debug: labels", labels.shape)

    num_samples = clean_images.shape[0]
    psnr_noisy = torch.zeros(num_samples)
    psnr_denoised = torch.zeros(num_samples)
    ssim_noisy = torch.zeros(num_samples)
    ssim_denoised = torch.zeros(num_samples)

    for i in range(num_samples):
        psnr_noisy[i] = skimage.metrics.peak_signal_noise_ratio(clean_images[i,0].numpy(), noisy_set[i,0].numpy())
        psnr_denoised[i] = skimage.metrics.peak_signal_noise_ratio(clean_images[i,0].numpy(), denoised_set[i,0].numpy())
        ssim_noisy[i] = skimage.metrics.structural_similarity(clean_images[i,0].numpy(), noisy_set[i,0].numpy(), data_range = 1.0)
        ssim_denoised[i] = skimage.metrics.structural_similarity(clean_images[i,0].numpy(), denoised_set[i,0].numpy(), data_range = 1.0)
        print("Processed sample ", i, " of ", num_samples)

    print("Finished Testing")

    return psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load denoised test set from saved file instead of raw ChestMNIST test
    denoised_images, denoised_labels = torch.load("chestmnist_test_denoised.pt")
    noisy_images, noisy_labels = torch.load("noisy_data/test_data.pt")
    dataset_test = ChestMNIST(split="test", download=True, size=28, transform=transform)

    #denoised_dataset = TensorDataset(denoised_images, denoised_labels)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

    psnr_noisy, psnr_denoised, ssim_noisy, ssim_denoised, labels = evaluate_model(test_loader, noisy_images, denoised_images, device)
    psnr_noisy_mean = torch.mean(psnr_noisy)
    psnr_denoised_mean = torch.mean(psnr_denoised)
    ssim_noisy_mean = torch.mean(ssim_noisy)
    ssim_denoised_mean = torch.mean(ssim_denoised)

    psnr_noisy_std = torch.std(psnr_noisy)
    psnr_denoised_std = torch.std(psnr_denoised)
    ssim_noisy_std = torch.std(ssim_noisy)
    ssim_denoised_std = torch.std(ssim_denoised)

    print("PSNR noisy: ", psnr_noisy_mean.numpy(), "+/-", psnr_noisy_std.numpy())
    print("PSNR denoised: ", psnr_denoised_mean.numpy(), "+/-", psnr_denoised_std.numpy())
    print("SSIM noisy: ", ssim_noisy_mean.numpy(), "+/-", ssim_noisy_std.numpy())
    print("SSIM denoised: ", ssim_denoised_mean.numpy(), "+/-", ssim_denoised_std.numpy())

if __name__ == "__main__":
    main()
    
