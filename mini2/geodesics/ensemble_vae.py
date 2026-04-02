# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

M = 2  # Default latent dimension

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def curve_energy(interior_pts, z_start, z_end, decoder_net):
    """
    Compute the pull-back curve energy.

    E(c) = sum_i || f(z_{i+1}) - f(z_i) ||^2
    where f = decoder_net (the decoder mean).

    Parameters:
    interior_pts: [torch.Tensor] shape (num_t-2, latent_dim), requires_grad=True
    z_start: [torch.Tensor] shape (latent_dim,)
    z_end:   [torch.Tensor] shape (latent_dim,)
    decoder_net: [torch.nn.Module] the decoder network (maps z -> image mean)

    Returns:
    energy: [torch.Tensor] scalar
    """
    # (num_t, latent_dim)
    curve_pts = torch.cat(
        [
            z_start.unsqueeze(0),
            interior_pts,
            z_end.unsqueeze(0),
        ],
        dim=0,
    )  # (num_t, latent_dim)

    # (num_t, 1, 28, 28)
    decoded = decoder_net(curve_pts)

    # (num_t, 784)
    decoded_flat = decoded.reshape(decoded.shape[0], -1)

    # (num_t-1, 784)
    diffs = decoded_flat[1:] - decoded_flat[:-1]
    energy = (diffs**2).sum()
    return energy


def compute_geodesic(z_start, z_end, decoder_net, num_t=20, lr=1e-2, num_steps=500):
    """
    Compute the geodesic between z_start and z_end under the pull-back metric
    of decoder_net by minimizing the curve energy.

    Parameters:
    z_start: [torch.Tensor] shape (latent_dim,)
    z_end:   [torch.Tensor] shape (latent_dim,)
    decoder_net: [torch.nn.Module]
    num_t: [int] total number of curve points including endpoints
    lr: [float] Adam learning rate
    num_steps: [int] number of gradient steps

    Returns:
    curve_pts: [torch.Tensor] shape (num_t, latent_dim)
    """
    # (num_t-2,)
    t_vals = torch.linspace(0, 1, num_t)[1:-1]
    interior_pts = (
        (
            z_start.unsqueeze(0) * (1 - t_vals.unsqueeze(1))
            + z_end.unsqueeze(0) * t_vals.unsqueeze(1)
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )

    optimizer = torch.optim.Adam([interior_pts], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        energy = curve_energy(interior_pts, z_start, z_end, decoder_net)
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        curve_pts = torch.cat(
            [
                z_start.unsqueeze(0),
                interior_pts,
                z_end.unsqueeze(0),
            ],
            dim=0,
        )
    return curve_pts


def curve_energy_ensemble(interior_pts, z_start, z_end, decoder_nets, ks, ls):
    """
    Compute the pull-back curve energy.

    E(c) = sum_i || f(z_{i+1}) - f(z_i) ||^2
    where f = decoder_net (the decoder mean).

    Parameters:
    interior_pts: [torch.Tensor] shape (num_t-2, latent_dim), requires_grad=True
    z_start: [torch.Tensor] shape (latent_dim,)
    z_end:   [torch.Tensor] shape (latent_dim,)
    decoder_nets: [list of torch.nn.Module] list of decoder networks in the ensemble
    ks, ls: [torch.Tensor] shape (S,) random indices for Monte Carlo sampling of pairs of decoders

    Returns:
    energy: [torch.Tensor] scalar
    """
    curve_pts = torch.cat(
        [
            z_start.unsqueeze(0),
            interior_pts,
            z_end.unsqueeze(0),
        ],
        dim=0,
    )  # (num_t, latent_dim)

    # Cache: only call each unique decoder once
    cache = {}
    for idx in set(ks + ls):
        cache[idx] = decoder_nets[idx](curve_pts).reshape(curve_pts.shape[0], -1)  # (T, 784)

    # Vectorized diff computation
    S = len(ks)
    out_k = torch.stack([cache[k] for k in ks])  # (S, T, 784)
    out_l = torch.stack([cache[l] for l in ls])  # (S, T, 784)
    diffs = out_k[:, 1:] - out_l[:, :-1]         # (S, T-1, 784)
    energy = (diffs ** 2).sum() / S

    return energy


def compute_geodesic_ensemble(
    z_start, z_end, decoder_nets, S=4, num_t=20, lr=1e-2, num_steps=500
):
    """
    Compute the geodesic between z_start and z_end under the pull-back metric of an ensemble of decoder_nets by minimizing the average curve energy.
    Parameters:
    z_start: [torch.Tensor] shape (latent_dim,)
    z_end:   [torch.Tensor] shape (latent_dim,)
    decoder_nets: [list of torch.nn.Module] list of decoder networks in the ensemble
    S: [int] number of Monte Carlo samples to use for approximating the average curve energy
    num_t: [int] total number of curve points including endpoints
    lr: [float] Adam learning rate
    num_steps: [int] number of gradient steps
    """
    device = next(decoder_nets[0].parameters()).device
    z_start = z_start.to(device)
    z_end = z_end.to(device)

    t_vals = torch.linspace(0, 1, num_t, device=device)[1:-1]
    interior_pts = (
        (
            z_start.unsqueeze(0) * (1 - t_vals.unsqueeze(1))
            + z_end.unsqueeze(0) * t_vals.unsqueeze(1)
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )

    optimizer = torch.optim.Adam([interior_pts], lr=lr)
    M = len(decoder_nets)

    for _ in range(num_steps):
        optimizer.zero_grad()

        ks, ls = (
            np.random.choice(M, size=(S,)).tolist(),
            np.random.choice(M, size=(S,)).tolist(),
        )
        energy = curve_energy_ensemble(
            interior_pts, z_start, z_end, decoder_nets, ks, ls
        )
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        curve_pts = torch.cat(
            [
                z_start.unsqueeze(0),
                interior_pts,
                z_end.unsqueeze(0),
            ],
            dim=0,
        )
    return curve_pts


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model.to(device)
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break


def new_decoder():
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net


def train_ensemble(
    model,
    optimizer,
    data_loader,
    epochs_per_decoder,
    device,
    num_decoders,
):
    # Train encoder and 1st decoder
    train(model, optimizer, data_loader, epochs_per_decoder, device)
    decoder_nets = [deepcopy(model.decoder.decoder_net)]
    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    for _ in tqdm(
        range(num_decoders - 1),
        desc="Training ensemble decoders",
        total=num_decoders - 1,
    ):
        # Add new decoder to the ensemble
        model.decoder = GaussianDecoder(new_decoder()).to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        # Train new decoder
        train(model, optimizer, data_loader, epochs_per_decoder, device)
        decoder_nets.append(deepcopy(model.decoder.decoder_net))

    return model, decoder_nets

def compute_curve_length_ensemble(curve_pts, decoder_nets):
    """
    Computes the continuous curve length in the observation space 
    under the exact model-average metric.
    L(c) = sum_i sqrt( E_{l,k} [||f_l(c_i) - f_k(c_{i+1})||^2] )
    """
    total_length = 0.0
    
    with torch.no_grad():
        cache = []
        for net in decoder_nets:
            decoded = net(curve_pts)
            cache.append(decoded.reshape(decoded.shape[0], -1))
        out_all = torch.stack(cache) # (num_decs, num_pts, 784)
        
        num_pts = curve_pts.shape[0]
        for i in range(num_pts - 1):
            f_l = out_all[:, i, :] # (num_decs, 784)
            f_k = out_all[:, i+1, :] # (num_decs, 784)
            
            diffs = f_l.unsqueeze(1) - f_k.unsqueeze(0) # (num_decs, num_decs, 784)
            
            sq_dists = (diffs**2).sum(dim=2)
            
            expected_sq_dist = sq_dists.mean()
            
            step_length = torch.sqrt(expected_sq_dist).item()
            total_length += step_length
            
    return total_length


def run_cov_evaluation(args, device, M, train_loader, test_loader):
    """
    Trains M=10 models, fixes 10 observation pairs, and calculates the CoV 
    for Euclidean and Geodesic distances using 1 to args.num_decoders.
    """
    
    all_models_encoders = []
    all_models_decoders = []
    
    num_retrainings = args.num_reruns if hasattr(args, 'num_reruns') else 10
    print(f"Starting {num_retrainings} VAE Retrainings")
    
    for m in range(num_retrainings):
        print(f"\n>> Training Model {m+1}/{num_retrainings}")
        
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model, decoder_nets = train_ensemble(
            model,
            optimizer,
            train_loader,
            args.epochs_per_decoder,
            device,
            args.num_decoders,
        )
        
        model.eval()
        for net in decoder_nets:
            net.eval()
            
        all_models_encoders.append(model.encoder)
        all_models_decoders.append(decoder_nets)
        
    # Fix point pairs from the test data
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch.to(device)
    
    num_pairs = args.num_curves if hasattr(args, 'num_curves') else 10
    torch.manual_seed(42)
    idx = torch.randperm(x_batch.size(0))[:num_pairs * 2]
    pairs = idx.reshape(num_pairs, 2)
    
    fixed_pairs_data = [(x_batch[a], x_batch[b]) for a, b in pairs]
    
    K_values = list(range(1, args.num_decoders + 1))
    avg_cov_euclidean = []
    avg_cov_geodesic = []
    
    # Evaluate distances
    for K in K_values:
        print(f"\nEvaluating for K={K} decoders...")
        pair_cov_euclidean = []
        pair_cov_geodesic = []
        
        for pair_idx, (y_i, y_j) in enumerate(fixed_pairs_data):
            d_euclidean = []
            d_geodesic = []
            
            for m in range(num_retrainings):
                encoder = all_models_encoders[m]
                active_decoders = all_models_decoders[m][:K]
                
                with torch.no_grad():
                    z_i = encoder(y_i.unsqueeze(0)).mean.squeeze(0)
                    z_j = encoder(y_j.unsqueeze(0)).mean.squeeze(0)
                
                # Euclidean Distance in latent space
                d_euclidean.append(torch.norm(z_i - z_j).item())
                
                # Geodesic Distance
                curve = compute_geodesic_ensemble(
                    z_i, z_j, active_decoders, 
                    S=args.monte_carlo_samples, 
                    num_t=args.num_t, 
                    lr=1e-2, 
                    num_steps=500
                )
                
                length = compute_curve_length_ensemble(curve, active_decoders)
                d_geodesic.append(length)
            
            # Compute CoV across the retrained models
            cov_e = np.std(d_euclidean) / np.mean(d_euclidean) if np.mean(d_euclidean) > 0 else 0
            cov_g = np.std(d_geodesic) / np.mean(d_geodesic) if np.mean(d_geodesic) > 0 else 0
            
            pair_cov_euclidean.append(cov_e)
            pair_cov_geodesic.append(cov_g)
            
        # Store average CoV across all pairs
        avg_cov_euclidean.append(np.mean(pair_cov_euclidean))
        avg_cov_geodesic.append(np.mean(pair_cov_geodesic))
        
    return K_values, avg_cov_euclidean, avg_cov_geodesic

def plot_cov_results(K_values, avg_cov_euclidean, avg_cov_geodesic, num_pairs, save_dir):
    """
    Plots the final CoV results as a function of the number of decoders.
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(K_values, avg_cov_euclidean, marker='o', label='Euclidean Distance', linestyle='--', color='blue')
    ax.plot(K_values, avg_cov_geodesic, marker='s', label='Geodesic Distance', linestyle='-', color='red')
    
    ax.set_title("Coefficient of Variation (CoV) vs. Ensemble Decoders")
    ax.set_xlabel("Number of Ensemble Decoders")
    ax.set_ylabel(f"Average CoV (across {num_pairs} pairs)")
    ax.set_xticks(K_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(save_dir, "cov_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved CoV plot to {out_path}")
    plt.show()

def new_encoder():
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )
    return encoder_net

if __name__ == "__main__":
    # Parse arguments
    import argparse

    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=[
            "train",
            "sample",
            "eval",
            "geodesics",
            "train_ensemble",
            "geodesics_ensemble",
            "evaluate_cov",
        ],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--monte-carlo-samples",
        type=int,
        default=4,
        metavar="N",
        help="number of Monte Carlo samples for ensemble geodesic (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def load_trained_model(model_path):
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def compute_latent_points(model):
        all_z = []
        all_y = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                # (batch, 2)
                z_mean = model.encoder(x).mean
                all_z.append(z_mean.cpu())
                all_y.append(y.cpu())
        # (N, 2)
        all_z = torch.cat(all_z, dim=0)
        # (N,)
        all_y = torch.cat(all_y, dim=0)

        return all_z, all_y

    # Choose mode to run
    if args.mode == "train":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "train_ensemble":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model, decoder_nets = train_ensemble(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
            args.num_decoders,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

        for i, decoder_net in enumerate(decoder_nets):
            torch.save(
                decoder_net.state_dict(),
                f"{experiments_folder}/decoder_{i}.pt",
            )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = load_trained_model(args.experiment_folder + "/model.pt")

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        model = load_trained_model(args.experiment_folder + "/model.pt")
        all_z, all_y = compute_latent_points(model)

        num_pairs = args.num_curves
        torch.manual_seed(42)
        idx = torch.randperm(len(all_z))[: num_pairs * 2]
        pairs = idx.reshape(num_pairs, 2)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            all_z[:, 0].numpy(),
            all_z[:, 1].numpy(),
            c=all_y.numpy(),
            cmap="tab10",
            s=5,
            alpha=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="class")

        decoder_net = model.decoder.decoder_net.cpu()
        decoder_net.eval()

        print(f"Computing {num_pairs} geodesics...")
        for i, (a, b) in enumerate(pairs):
            z_start = all_z[a]
            z_end = all_z[b]
            curve = compute_geodesic(
                z_start, z_end, decoder_net, num_t=args.num_t, lr=1e-2, num_steps=500
            )
            ax.plot(
                curve[:, 0].numpy(),
                curve[:, 1].numpy(),
                color="black",
                linewidth=0.8,
                alpha=0.7,
            )
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{num_pairs} done")

        ax.set_title("Latent space with pull-back geodesics (Part A)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        plt.tight_layout()

        out_path = os.path.join(args.experiment_folder, "geodesics_partA.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    elif args.mode == "geodesics_ensemble":
        decoder_nets = []
        for i in range(args.num_decoders):
            decoder_net = new_decoder().to(device)
            decoder_net.load_state_dict(
                torch.load(
                    args.experiment_folder + f"/decoder_{i}.pt", map_location=device
                )
            )
            decoder_net.eval()
            decoder_nets.append(decoder_net)

        model = load_trained_model(args.experiment_folder + "/model.pt")
        all_z, all_y = compute_latent_points(model)

        num_pairs = args.num_curves
        torch.manual_seed(42)
        idx = torch.randperm(len(all_z))[: num_pairs * 2]
        pairs = idx.reshape(num_pairs, 2)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            all_z[:, 0].numpy(),
            all_z[:, 1].numpy(),
            c=all_y.numpy(),
            cmap="tab10",
            s=5,
            alpha=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="class")

        print(f"Computing {num_pairs} ensemble geodesics...")
        for i, (a, b) in enumerate(pairs):
            z_start = all_z[a]
            z_end = all_z[b]
            curve = compute_geodesic_ensemble(
                z_start,
                z_end,
                decoder_nets,
                S=args.monte_carlo_samples,
                num_t=args.num_t,
                lr=1e-2,
                num_steps=500,
            )
            curve_cpu = curve.detach().cpu()
            ax.plot(
                curve_cpu[:, 0].numpy(),
                curve_cpu[:, 1].numpy(),
                color="black",
                linewidth=0.8,
                alpha=0.7,
            )
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{num_pairs} done")

        ax.set_title("Latent space with ensemble pull-back geodesics (Part B)")
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        plt.tight_layout()

        out_path = os.path.join(args.experiment_folder, "geodesics_partB.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")

    elif args.mode == "evaluate_cov":
        print("Starting CoV Evaluation...")
        os.makedirs(args.experiment_folder, exist_ok=True)
        
        K_values, avg_cov_e, avg_cov_g = run_cov_evaluation(
            args=args, 
            device=device, 
            M=M, 
            train_loader=mnist_train_loader, 
            test_loader=mnist_test_loader
        )

        plot_cov_results(
            K_values, 
            avg_cov_e, 
            avg_cov_g, 
            num_pairs=args.num_curves, 
            save_dir=args.experiment_folder
        )
