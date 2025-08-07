import torch
from torchvision import transforms
from medmnist import ChestMNIST
import gmi
import os
from tqdm import tqdm

#output directory
SAVE_DIR = "noisy_data"
os.makedirs(SAVE_DIR, exist_ok=True)

#transformation (convert pil to tensor)
#PIL/pillow = python imaging library fork that provides image processing capabilities
transform = transforms.ToTensor()

#gaussian noise settings
noise_std = 0.1
noise_adder = gmi.distribution.AdditiveWhiteGaussianNoise(noise_standard_deviation=noise_std)

#splits to process
splits = ["train","val","test"]

for split in splits:
    print(f"\nProcessing split: {split}")

    #load clean dataset split
    dataset = ChestMNIST(split=split,download=True,transform=transform)

    noisy_imgs = []
    labels = []

    for i in tqdm(range(len(dataset)), desc=f"{split} split"):
        img, label = dataset[i] #img: [1, 28, 28], label: [14]. label is a multi-hot vector of size 14
        
        img = img.unsqueeze(0) #shape: [1, 1, 28, 28]
        noisy_img = noise_adder(img).squeeze(0) #back to [1, 28, 28]

        label_tensor = torch.tensor(label)
        if i < 3:  # Optional preview
            print(f"Sample {i} label: {label_tensor}")

        noisy_imgs.append(noisy_img)
        labels.append(label_tensor)

    #stack into tensors
    noisy_imgs = torch.stack(noisy_imgs) #shape: [N, 1, 28, 28]
    labels = torch.stack(labels).float() #shape: [N, 14]

    #save both together in one .pt file
    save_path = os.path.join(SAVE_DIR, f"{split}_data.pt")
    torch.save({'images': noisy_imgs, 'labels': labels}, save_path)

    print(f"Saved noisy {split} data to {save_path}")

print("\nAll splits processed and saved successfully.")
