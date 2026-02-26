import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import create_vae_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training') 
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable')

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

    if args.latent_dim != 2:
        print("Plotting requires latent-dim=2.")
        exit()
        
    # Define VAE model
    model = create_vae_model(args.prior, args.latent_dim, device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model.pt")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    
    # Gather aggregate posterior samples
    z_posterior = []
    with torch.no_grad():
        for x in mnist_test_loader:
            x = x[0].to(device)
            q = model.encoder(x)
            z_posterior.append(q.sample().cpu())
    z_posterior = torch.cat(z_posterior, dim=0).numpy()
    
    # Evaluate Prior Density Grid
    grid_x, grid_y = torch.meshgrid(torch.linspace(-5, 5, 100), torch.linspace(-5, 5, 100), indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
    
    with torch.no_grad():
        log_prob = model.prior.log_prob(grid)
        prob = torch.exp(log_prob).view(100, 100).cpu().numpy()
        
    plt.figure(figsize=(10, 8))
    plt.contour(grid_x.numpy(), grid_y.numpy(), prob, levels=20, cmap='Blues', linewidths=2)
    plt.scatter(z_posterior[:, 0], z_posterior[:, 1], alpha=0.15, s=5, c='red', edgecolors='none', label='Aggregate Posterior')
    plt.legend()
    plt.title(f'{args.prior.upper()} Prior and Aggregate Posterior')
    sample_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_sample_plot.png")
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    plt.savefig(sample_path)
    plt.close()
