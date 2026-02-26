import torch
import argparse
import os
from torchvision import datasets, transforms
from vae import train, evaluate
from model import create_vae_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
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

    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True
    )

    # Define VAE model
    model = create_vae_model(args.prior, args.latent_dim, device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model.pt")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, mnist_train_loader, args.epochs, args.device)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Evaluate after training
    test_elbo = evaluate(model, mnist_test_loader, device)
    print(f"Test Set ELBO: {test_elbo:.4f}")
