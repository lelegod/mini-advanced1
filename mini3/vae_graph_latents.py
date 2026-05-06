import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Constants for MUTAG dataset
MAX_NODES = 28  # Maximum nodes in a MUTAG graph
EDGE_PROBABILITY_CUTOFF = 0.22
TEMPERATURE = 3

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(self.norm(h))
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        
        hg = global_mean_pool(h, batch) 
        mu = self.fc_mu(hg)
        logstd = self.fc_logstd(hg)
        return mu, logstd

class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, max_nodes * max_nodes)
        )

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, temperature=1.0):
        logits = self.decoder(z) / temperature
        return logits.view(-1, self.max_nodes, self.max_nodes)

class GraphLatentVAEGenerator:
    def __init__(self, model):
        self.model = model

    def forward(self, num_samples=1):
        self.model.eval()
        generated_graphs = []
        with torch.no_grad():
            # Sample from latent prior
            z = torch.randn(num_samples, self.model.encoder.fc_mu.out_features)
            logits = self.model.decode(z, temperature=TEMPERATURE)
            prob_adj = torch.sigmoid(logits)

            for i in range(num_samples):
                adj = prob_adj[i]
                adj = (adj + adj.t()) / 2
                adj.fill_diagonal_(0)
                
                # Create binary adjacency
                bin_adj = (adj > EDGE_PROBABILITY_CUTOFF).float()
                
                # --- PRUNING LOGIC ---
                # Find nodes that have at least one connection
                degrees = bin_adj.sum(dim=1)
                active_nodes = (degrees > 0).nonzero(as_tuple=True)[0]
                
                if len(active_nodes) > 0:
                    # Filter the adjacency matrix to only active nodes
                    pruned_adj = bin_adj[active_nodes][:, active_nodes]
                    edge_index = pruned_adj.nonzero(as_tuple=False).t().contiguous()
                    num_nodes = len(active_nodes)
                else:
                    # Fallback for empty graphs: just a single isolated node
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    num_nodes = 1

                generated_graphs.append(Data(edge_index=edge_index, num_nodes=num_nodes))
        return generated_graphs

def train_vae(train_dataset, model, epochs=300):
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4) # Slightly slower LR

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        avg_recon = 0
        avg_kl = 0
        avg_loss = 0
        for data in loader:
            opt.zero_grad()
            mu, logstd = model.encoder(data.x, data.edge_index, data.batch)
            z = model.reparameterize(mu, logstd)
            logits = model.decode(z, temperature=1.0) # Train at T=1
            
            # Prepare Target
            adj_dense = to_dense_adj(data.edge_index, data.batch)
            batch_size = adj_dense.size(0)
            target_adj = torch.zeros(batch_size, MAX_NODES, MAX_NODES).to(adj_dense.device)
            target_adj[:, :adj_dense.size(1), :adj_dense.size(2)] = adj_dense

            # Calculate Positional Weight (ratio of zeros to ones)
            # This helps the model stop being "lazy" and actually predict edges.
            pos_weight = (MAX_NODES * MAX_NODES - target_adj.sum()) / (target_adj.sum() + 1e-6)
            pos_weight = torch.clamp(pos_weight, 1.0, 10.0) # Keep it reasonable

            recon_loss = F.binary_cross_entropy_with_logits(
                logits, target_adj, pos_weight=pos_weight
            )
            
            kl_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd), dim=1))
            
            # KL Annealing
            beta = min(0.05, epoch / 500) 
            loss = recon_loss + (beta * kl_loss)
            
            loss.backward()
            opt.step()

            avg_recon += recon_loss.item()
            avg_kl += kl_loss.item()
            avg_loss += loss.item()

        if epoch % 20 == 0:
            n = len(loader)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss/n:.4f} | Recon: {avg_recon/n:.4f} | KL: {avg_kl/n:.4f} | Beta: {beta:.3f}")

def train(model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    model = GraphVAE(
        in_channels=dataset.num_features,
        hidden_dim=64,
        latent_dim=32,
        max_nodes=MAX_NODES
    )

    train_vae(train_dataset, model)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def sample(num_samples, model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")
    model = GraphVAE(
        in_channels=dataset.num_features,
        hidden_dim=64,
        latent_dim=32,
        max_nodes=MAX_NODES
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model.load_state_dict(torch.load(model_path))
    generator = GraphLatentVAEGenerator(model)
    graphs = generator.forward(num_samples)

    output_path = os.path.join("mini3", "generated_graphs_vae.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    print(f"Generated {len(graphs)} graphs and saved to {output_path}.")

def main():
    torch.set_printoptions(precision=5, sci_mode=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vae_graph_latent.pt", help="Path to model weights.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("num_samples", type=int)

    args = parser.parse_args()
    if args.command == "train":
        train(args.model)
    elif args.command == "sample":
        sample(args.num_samples, args.model)

if __name__ == "__main__":
    main()