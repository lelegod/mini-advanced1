import torch
import torch.nn as nn
from flow import FlowPrior
from vae import BernoulliDecoder, GaussianEncoder, VAE, GaussianPrior, MoGPrior

def create_vae_model(prior_type, M, device):
    """
    Creates and returns a VAE model with the specified prior.
    
    Parameters:
    prior_type: [str]
        The type of prior to use ('gaussian' or 'flow')
    M: [int]
        The latent dimension
    device: [str or torch.device]
        The device to place the model on
    """
    if prior_type == 'gaussian':
        prior = GaussianPrior(M)
    elif prior_type == 'mog':
        prior = MoGPrior(M, num_components=10)
    elif prior_type == 'flow':
        prior = FlowPrior(M, num_transformations=5, num_hidden=64)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)
    
    return model
