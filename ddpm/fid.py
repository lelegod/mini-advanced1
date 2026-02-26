import numpy as np
import scipy
import torch
from ddpm import DDPM, ImageOnlyDataset
from unet import Unet
import argparse
from torchvision import datasets, transforms
from time import perf_counter

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = torch.nn.Sequential(

            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.25),

            torch.nn.Flatten(),
            torch.nn.Linear(9216, 128),
            torch.nn.Dropout(0.5),
        )

        self.classification_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        y = self.layers(x)
        y = self.classification_layer(y)
        return y
    

def frechet_distance(x_a, x_b):
    mu_a = np.mean(x_a, axis=0)
    sigma_a = np.cov(x_a.T)
    mu_b = np.mean(x_b, axis=0)
    sigma_b = np.cov(x_b.T)

    diff = mu_a - mu_b
    covmean = scipy.linalg.sqrtm(sigma_a @ sigma_b)
    return np.sum(diff**2) + np.trace(sigma_a + sigma_b - 2.0 * covmean)


def compute_fid(
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    device: str = "cpu",
    classifier_ckpt: str = "mnist_classifier.pth",
) -> float:
    """Compute the Fréchet Inception Distance (FID) between two sets of images.
    
    Args:
        x_real (torch.Tensor): A batch of real images, shape (N, 1, 28, 28) and with values in [-1, 1].
        x_gen (torch.Tensor): A batch of generated images, shape (N, 1, 28, 28) and with values in [-1, 1].
        device (str): The device to run the classifier on ("cpu" or "cuda").
        classifier_ckpt (str): Path to the pre-trained classifier checkpoint

    Returns:
        float: The computed FID score between the two sets of images.
    """

    
    # ---- load classifier ----
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    clf.eval()

    # ---- calculate latent features with classifier ----
    with torch.no_grad():
        real_latent = clf.layers(x_real)
        gen_latent = clf.layers(x_gen)
    real_latent = real_latent.cpu().numpy()
    gen_latent = gen_latent.cpu().numpy()

    return frechet_distance(real_latent, gen_latent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate and compare.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the classifier on (cpu or cuda).")
    parser.add_argument("--cls", type=str, default="mnist_classifier.pth", help="Path to the pre-trained classifier checkpoint.")
    parser.add_argument("--model", type=str, default="ddpm_mnist.pth", help="Path to the pre-trained DDPM model checkpoint.")
    args = parser.parse_args()

    N = int(args.num_samples)

    #load dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        ]
    )
    train_data = ImageOnlyDataset(
        datasets.MNIST("data/", train=True, download=True, transform=transform)
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=N, shuffle=True
    )

    x_real = next(iter(train_loader)).to(args.device)

    # Define the network
    network = Unet()

    # Set the number of steps in the diffusion process
    T = 1000

    # Define model
    model = DDPM(network, T=T).to(args.device)

    model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )
    model.eval()
    with torch.no_grad():
        start = perf_counter()
        x_gen = (model.sample((N, 784))).cpu()
        end = perf_counter()

    x_gen = x_gen.view(-1, 1, 28, 28).to(args.device)

    print(f'x_gen shape: {x_gen.shape}')
    print(f'x_real shape: {x_real.shape}')
    print(f'x_gen min: {x_gen.min()}, max: {x_gen.max()}')
    print(f'x_real min: {x_real.min()}, max: {x_real.max()}')
    # Compute FID
    fid_score = compute_fid(x_real, x_gen, device=args.device, classifier_ckpt=args.cls)
    print(f"FID score: {fid_score}")
    print(f'Samples per second: {N / (end - start)}')

