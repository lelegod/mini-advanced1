import torch
import argparse
import os
import numpy as np
from torchvision import datasets, transforms
from vae import train, evaluate
from model import create_vae_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable')
    parser.add_argument('--runs', type=int, default=1, metavar='N', help='number of independent training runs')

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

    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False
    )

    # Define VAE model
    test_elbos = []
    for run in range(args.runs):
        print(f"Starting Run {run + 1}/{args.runs}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model.pt")

        model = create_vae_model(args.prior, args.latent_dim, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model_run{run}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        
        test_elbo = evaluate(model, mnist_test_loader, device)
        print(f"Run {run + 1} Test Set ELBO: {test_elbo:.4f}")
        test_elbos.append(test_elbo)

    # Print final statistics for the report
    print("Final Results")
    print(f"Prior: {args.prior.upper()} | Latent Dim: {args.latent_dim} | Epochs: {args.epochs}")
    if args.runs > 1:
        mean_elbo = np.mean(test_elbos)
        std_elbo = np.std(test_elbos)
        print(f"Test Set ELBO over {args.runs} runs: {mean_elbo:.4f} ± {std_elbo:.4f}")
    else:
        print(f"Test Set ELBO: {test_elbos[0]:.4f}")