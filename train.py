#DENOISER
from functools import partial
import torch
import gmi
import os
import torchvision.transforms as transforms
if __name__ == '__main__':
    #determine whether it should be placed on gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    medmnist_name = 'ChestMNIST'
    batch_size = 32

    #path of current script
    medmnist_example_dir = os.path.dirname(os.path.abspath(__file__))

    #define root directory for MedMNIST dataset
    DATA_ROOT = os.path.join(medmnist_example_dir, 'data')  # or just './data'

    #make sure the directory exists
    os.makedirs(DATA_ROOT, exist_ok=True)

    #labels discarded when dataset is created (target_transform set)
    dataset_train = gmi.datasets.MedMNIST(medmnist_name,
                                      split='train',
                                      download=True,
                                      images_only=True,
                                      root=DATA_ROOT)

    #validation dataset (monitor model's perf during training, in between epochs)
    dataset_val = gmi.datasets.MedMNIST(medmnist_name,
                                    split='val',
                                    download=True,
                                    images_only=True,
                                    root=DATA_ROOT)

    dataset_test = gmi.datasets.MedMNIST(medmnist_name,
                                        split='test',
                                        download=True,
                                        images_only=True,
                                        root=DATA_ROOT)

    #train dataloader: uses training set, batch size, shuffles data, only has 1 worker process
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=1)

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=1)

    #define simulator
    #noise standard deviation: tells how random it should be
    white_noise_adder = gmi.distribution.AdditiveWhiteGaussianNoise(
                                                    noise_standard_deviation=0.1)

    #denoiser: 1 input channel (black and white not rgb), 1 output channel(grayscale?), hidden channels is filters
    # activation siLU: activation function
    #dim: means that it operates on 2d images
    denoiser = gmi.network.SimpleCNN(input_channels=1,
                                    output_channels=1,
                                    hidden_channels_list=[16, 32, 64, 128, 256, 128, 64, 32, 16],
                                    activation=torch.nn.SiLU(),
                                    dim=2).to(device)

    #define denoising task
    mnist_denoising_task = gmi.tasks.ImageReconstructionTask(
                                        image_dataset = dataset_train,
                                        measurement_simulator = white_noise_adder,
                                        image_reconstructor = denoiser,
                                        device=device)

    #loss function: mean squared error
    loss_closure = mnist_denoising_task.loss_closure(torch.nn.MSELoss())

    #train denoiser
    #dataloader, loss function, epochs, iterations per epoch, no optimizer (why?)
    #learning rate (how drastic of adjustment of weights), CPU
    # after every epoch the validation loader is used, 10 validation steps per epoch
    #verbose gives you feedback during training
    gmi.train(
        loss_closure=loss_closure,
        num_epochs=20,
        num_iterations=10,
        optimizer=None,
        lr=1e-3,
        device=device,
        validation_loader=dataloader_val,
        num_iterations_val=10,
        verbose=True
    )
    
    save_path = os.path.abspath('denoiser.pth')
    torch.save(denoiser.state_dict(), save_path)
    print(f"Denoiser weights saved to: {save_path}")
    

    #sample_images_measurements_reconstructions is a function; may be used to select images?
    #9 clean/original images selected
    #9 noisy images selected
    #9 reconstructed images selected
    images, measurements, reconstructions = mnist_denoising_task.sample_images_measurements_reconstructions(
                                    image_batch_size=9,
                                    measurement_batch_size=9,
                                    reconstruction_batch_size=9)

    from matplotlib import pyplot as plt

    images = images.cpu().detach().numpy()
    measurements = measurements.cpu().detach().numpy()
    reconstructions = reconstructions.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(images[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.savefig(medmnist_example_dir + '/images.png')

    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(measurements[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.savefig(medmnist_example_dir + '/measurements.png')

    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(reconstructions[0,0,i].transpose(1,2,0), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.savefig(medmnist_example_dir + '/reconstructions.png')
