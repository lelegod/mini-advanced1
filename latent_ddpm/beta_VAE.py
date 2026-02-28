import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import M, BETA_VAE, BATCH_SIZE, TRAIN_SIZE, VAE_EPOCHS, VAE_PATIENCE, VAE_DECODER_STD, VAE_LR

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01): 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"   -> EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

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
        mean = self.decoder_net(z)
        std = torch.ones_like(mean) * VAE_DECODER_STD 
        
        return td.Independent(td.Normal(loc=mean, scale=std), 1)

class BetaVAE(nn.Module):
    def __init__(self, prior, decoder, encoder, beta=None):
        super(BetaVAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta if beta is not None else BETA_VAE

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - self.beta * td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def forward(self, x):
        return -self.elbo(x)

def train_vae(model, optimizer, train_loader, val_loader, epochs, device, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for x, _ in progress_bar:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                loss = model(x)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"End of Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}! Validation loss stopped improving.")
            break

def get_vae_model(M=None, beta=None, device='cpu'):
    """Instantiate the VAE model architecture."""
    if M is None:
        M = globals()['M']
    if beta is None:
        beta = globals()['BETA_VAE']
        
    encoder_net = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Tanh()
    )
    prior = GaussianPrior(M)
    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    return BetaVAE(prior, decoder, encoder, beta=beta).to(device)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = BATCH_SIZE
    epochs = VAE_EPOCHS
    beta = BETA_VAE

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])

    # Load the full dataset
    full_train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    
    # Split into training and validation images
    train_size = TRAIN_SIZE
    val_size = len(full_train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size])

    # Create loaders for both
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    vae_model = get_vae_model(M, beta, device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=VAE_LR)
    train_vae(vae_model, optimizer, train_loader, val_loader, epochs, device, patience=VAE_PATIENCE)

    torch.save(vae_model.state_dict(), 'beta_vae.pt')
    print("Saved Beta-VAE weights to beta_vae.pt")

    vae_model.eval()
    with torch.no_grad():
        z_sample = torch.randn(64, M).to(device)
        generated_dist = vae_model.decoder(z_sample)
        generated_images = generated_dist.mean.view(-1, 1, 28, 28)
        save_image(generated_images, 'vae_samples.png', nrow=8, normalize=True, value_range=(-1, 1))
        print("Saved VAE samples to vae_samples.png")