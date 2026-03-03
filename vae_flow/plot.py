from matplotlib.pyplot import legend
import torch
import torch.nn as nn
import argparse
import os
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import create_vae_model
from sklearn.decomposition import PCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training') 
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load binarized MNIST
    threshold = 0.5
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: (threshold < x).float().squeeze())
    ])

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False
    )

    # Define VAE model
    model = create_vae_model(args.prior, args.latent_dim, device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model_run0.pt")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    
    # Gather aggregate posterior samples
    z_posterior = []
    labels = []
    with torch.no_grad():
        for x, y in mnist_test_loader:
            x = x.to(device)
            q = model.encoder(x)
            z_posterior.append(q.sample().cpu())
            labels.append(y)
    
    z_posterior = torch.cat(z_posterior, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # Sample from the Prior
    with torch.no_grad():
        z_prior = model.prior().sample((10000,)).cpu().numpy()

    # Fit PCA on the Posterior and Transform both
    pca = PCA(n_components=2)
    z_post_2d = pca.fit_transform(z_posterior)
    z_prior_2d = pca.transform(z_prior)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    plt.scatter(z_post_2d[:, 0], z_post_2d[:, 1], s=4, c='gray', edgecolors='none')
    sns.kdeplot(x=z_prior_2d[:, 0], y=z_prior_2d[:, 1], levels=10, color='blue', alpha=0.7, linewidths=1.5)
    
    prior_handle = mlines.Line2D([], [], color='blue', linewidth=2, label='Prior Density')
    agg_post_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='gray', markersize=6, label='Aggregate Posterior')
    
    plt.legend(handles=[prior_handle, agg_post_handle], loc='upper right', fontsize=30)
    plt.title(f'{args.prior.upper()} Prior (PCA, M={args.latent_dim})', fontsize=45)
    plt.xlabel('Principal Component 1', fontsize=30)
    # plt.ylabel('Principal Component 2', fontsize=30)
    plt.tight_layout()
    sample_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_run0_sample_plot.png")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    plt.savefig(sample_path, dpi=150)
    plt.close()
    
    print(f"Explained variance ratio of first 2 PCs: {pca.explained_variance_ratio_}")