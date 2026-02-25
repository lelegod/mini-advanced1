import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm

# VAE Components
class GaussianPrior(nn.Module):
    def __init__(self, M):
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean, std = torch.chunk(self.decoder_net(z), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class BetaVAE(nn.Module):
    def __init__(self, prior, decoder, encoder, beta=1.0):
        super(BetaVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def forward(self, x):
        return -self.elbo(x)

# Training Loop 
def train_vae(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training Beta-VAE")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

# Main Execution
def get_vae_model(M, beta, device):
    """Helper function to instantiate the model architecture."""
    encoder_net = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784*2),
    )
    prior = GaussianPrior(M)
    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    return BetaVAE(prior, decoder, encoder, beta=beta).to(device)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 64
    epochs = 3 

    # Latent dimension
    M = 32 

    beta = 1e-6  

    # Transformations from Week 3 exercises
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])

    train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize and train
    vae_model = get_vae_model(M, beta, device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-6)
    train_vae(vae_model, optimizer, train_loader, epochs, device)

    # EXPORT WEIGHTS
    torch.save(vae_model.state_dict(), 'beta_vae.pt')
    print("Saved Beta-VAE weights to beta_vae.pt")