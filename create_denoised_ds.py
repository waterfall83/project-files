import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import gmi

#config
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISY_DIR = 'noisy_data'
OUTPUT_DIR = 'denoised_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

#load trained denoiser
denoiser = gmi.network.SimpleCNN(
    input_channels=1,
    output_channels=1,
    hidden_channels_list=[16, 32, 64, 128, 256, 128, 64, 32, 16],
    activation=torch.nn.SiLU(),
    dim=2
).to(DEVICE)

denoiser.load_state_dict(torch.load('denoiser.pth', map_location=DEVICE))
denoiser.eval()
print("Loaded trained denoiser.")

#helper function
def denoise_and_save(input_path, output_path):
    print(f"\nProcessing {input_path}...")

    #file existence check
    if not os.path.exists(input_path):
        print(f"Skipping missing file: {input_path}")
        return

    data = torch.load(input_path)
    images, labels = data['images'], data['labels']

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    denoised_images = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Denoising {os.path.basename(input_path)}"):
            noisy_imgs = batch[0].to(DEVICE)
            denoised = denoiser(noisy_imgs)
            denoised_images.append(denoised.cpu())
    
    denoised_images = torch.cat(denoised_images)
    torch.save({'images': denoised_images, 'labels': labels}, output_path)
    print(f"Saved denoised data to {output_path}")
    #log number of samples saved
    print(f"Saved {len(denoised_images)} samples.")

#run on all noisy sets
for split in ['train', 'val', 'test']:
    input_file = os.path.join(NOISY_DIR, f"{split}_data.pt")
    output_file = os.path.join(OUTPUT_DIR, f"{split}_denoised.pt")
    denoise_and_save(input_file, output_file)

print("\nAll noisy datasets denoised and saved.")
