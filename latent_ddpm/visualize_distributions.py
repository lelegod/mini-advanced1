import torch
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd

from config import M, BETA_VAE, NUM_HIDDEN, T
from beta_VAE import get_vae_model 
from ddpm import DDPM, FcNetwork 

def plot_latent_distributions():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    m = M
    beta = BETA_VAE
    num_hidden = NUM_HIDDEN
    t = T
    num_samples = 2000

    # Load VAE and DDPM
    vae_model = get_vae_model(m, beta, device)
    vae_model.load_state_dict(torch.load('beta_vae.pt', map_location=device))
    vae_model.eval()

    network = FcNetwork(input_dim=m, num_hidden=num_hidden)
    latent_ddpm = DDPM(network, T=t).to(device)
    latent_ddpm.load_state_dict(torch.load('latent_ddpm.pt', map_location=device))
    latent_ddpm.eval()

    # Get Aggregate Posterior Samples (from real data)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])
    test_data = datasets.MNIST('data/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=num_samples, shuffle=True)
    
    real_x = next(iter(test_loader))[0].to(device)
    stats = torch.load('latent_stats.pt', map_location=device)
    with torch.no_grad():
        z_posterior = vae_model.encoder(real_x).sample().cpu().numpy()

    with torch.no_grad():
        z_ddpm_std = latent_ddpm.sample((num_samples, M)).to(device)
        z_ddpm = (z_ddpm_std * stats['std']) + stats['mean']
        z_ddpm = z_ddpm.cpu().numpy()

    z_prior = torch.randn(num_samples, M).numpy()

    # Fit PCA on the aggregate posterior to establish the main axes of variance
    pca = PCA(n_components=2)
    pca.fit(z_posterior)

    z_posterior_2d = pca.transform(z_posterior)
    z_ddpm_2d = pca.transform(z_ddpm)
    z_prior_2d = pca.transform(z_prior)

    # Create DataFrames for Seaborn
    df_post = pd.DataFrame(z_posterior_2d, columns=['PC1', 'PC2'])
    df_ddpm = pd.DataFrame(z_ddpm_2d, columns=['PC1', 'PC2'])

    plt.figure(figsize=(10, 8))
    
    plt.scatter(z_prior_2d[:, 0], z_prior_2d[:, 1], 
                alpha=0.15, s=10, color='gray', label='VAE Prior $p(z)=\mathcal{N}(0,I)$')
    sns.kdeplot(data=df_post, x='PC1', y='PC2', 
                color='blue', levels=6, linewidths=2.5, 
                label='Aggregate Posterior $q_{\phi}(z)$')
    sns.kdeplot(data=df_ddpm, x='PC1', y='PC2', 
                color='red', levels=6, linewidths=2.5, linestyles='--', 
                label='Latent DDPM $p_{\\theta}(z)$')

    plt.title(f'Latent Space Comparison (PCA, M={m})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='VAE Prior $p(z)$'),
        Line2D([0], [0], color='blue', lw=2.5, label='Aggregate Posterior $q_{\phi}(z)$'),
        Line2D([0], [0], color='red', lw=2.5, linestyle='--', label='Latent DDPM $p_{\\theta}(z)$')
    ]
    plt.legend(handles=handles, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('latent_density_comparison.png')
    print("Saved labeled plot to latent_density_comparison.png")

if __name__ == "__main__":

    plot_latent_distributions()