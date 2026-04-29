import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        return mu, logstd


class NodeVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        prob_adj = torch.sigmoid(z @ z.t())
        return prob_adj

import numpy as np
from torch_geometric.data import Data
from collections import defaultdict


class NodeLatentVAEGenerator:
    def __init__(self, train_dataset, model):
        self.train_dataset = train_dataset
        self.model = model
        self.node_counts = [g.num_nodes for g in train_dataset]

    def sample_num_nodes(self, num_samples=1):
        return np.random.choice(self.node_counts, size=num_samples)

    def sample_graph(self, N):
        graphs = []

        for n in N:
            # Sample node latents
            z = torch.randn(n, self.model.encoder.conv_mu.out_channels)

            # Decode adjacency
            prob_adj = torch.sigmoid(z @ z.t())

            # Nodes must not connect to thmeselves
            prob_adj.fill_diagonal_(0)

            adj = (prob_adj > 0.5).float()

            edge_index = adj.nonzero(as_tuple=False).t().contiguous()

            graphs.append(Data(edge_index=edge_index, num_nodes=n))

        return graphs

    def forward(self, num_samples=1):
        N = self.sample_num_nodes(num_samples)
        graphs = self.sample_graph(N)

        adj_matrices = [
            to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)
            .squeeze(0)
            .numpy()
            for g in graphs
        ]

        return graphs, adj_matrices

def train(model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")

    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    model = NodeVAE(
        in_channels=dataset.num_features,
        hidden_dim=32,
        latent_dim=16
    )

    train_vae(train_dataset, model)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

from torch_geometric.loader import DataLoader

def train_vae(train_dataset, model, epochs=50):
    # Use a DataLoader for stability
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Slightly lower LR

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()

            x, edge_index, batch = data.x, data.edge_index, data.batch
            mu, logstd = model.encoder(x, edge_index)
            z = model.reparameterize(mu, logstd)

            # Reconstruction: Inner product decoder
            logits = z @ z.t()
            
            # Get ground truth adjacency
            adj = to_dense_adj(edge_index, max_num_nodes=data.num_nodes).squeeze(0)

            # Create mask for all elements EXCEPT the diagonal
            mask = ~torch.eye(data.num_nodes, device=z.device).bool()
            #print(mask)

            pos_weight = (adj.numel() - adj.sum()) / (adj.sum())

            # Only apply loss to the actual edges (off-diagonal)
            recon_loss = F.binary_cross_entropy_with_logits(
                logits[mask], 
                adj[mask],
                pos_weight=pos_weight
            )
            #print(logits)
            #print(logits[mask])
            
            # Stabilized Loss
            #recon_loss = F.binary_cross_entropy_with_logits(logits, adj)

            # Refined KL Divergence
            # Formula: 0.5 * sum(exp(2*logstd) + mu^2 - 1 - 2*logstd)
            kl = -0.5 * torch.mean(torch.sum(1 + 2*logstd - mu**2 - torch.exp(2*logstd), dim=1))
            
            # Norm factor: helps scale KL relative to the number of nodes
            # A common trick in VGAE
            kl = kl / data.num_nodes 

            # Warm-up schedule: reach 1.0 faster (e.g., by epoch 20)
            beta = min(1.0, epoch / 20) 
            
            loss = recon_loss + beta * kl
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Avg Loss: {total_loss / len(train_loader):.4f}")

def sample(num_samples, model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")

    model = NodeVAE(
        in_channels=dataset.num_features,
        hidden_dim=32,
        latent_dim=16
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    generator = NodeLatentVAEGenerator(train_dataset, model)

    graphs, adj_matrices = generator.forward(num_samples)

    torch.save(graphs, ".\\mini3\\generated_graphs_vae.pt")

    print(f"Generated {len(graphs)} graphs.")

def main():
    parser = argparse.ArgumentParser()
    model_path = ".\\mini3\\vae_node_latent.pt"

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    subparsers.add_parser("train")

    # Sample command
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("num_samples", type=int)

    args = parser.parse_args()

    if args.command == "train":
        train(model_path)

    elif args.command == "sample":
        sample(args.num_samples, model_path)


if __name__ == "__main__":
    main()