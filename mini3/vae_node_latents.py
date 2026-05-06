import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset

LOGITS_DENOMINATOR = 4
EDGE_PROBABILITY_CUTOFF = 0.65

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        # Use LayerNorm to keep the scale of node features distinct
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        # 1. First convolution
        h = self.conv1(x, edge_index)
        h = self.norm(h)
        h = F.leaky_relu(h) # LeakyRelu helps prevent dead neurons
        
        # 2. Output layers
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        
        return mu, logstd


class NodeVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        
        # New MLP Decoder: Takes concat([zi, zj]) which has size latent_dim * 2
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        num_nodes = z.size(0)

        z_i = z.unsqueeze(1).expand(-1, num_nodes, -1)
        z_j = z.unsqueeze(0).expand(num_nodes, -1, -1)

        pair = torch.cat([z_i, z_j], dim=-1)  # (N, N, 2*latent)

        logits = self.decoder_mlp(pair).squeeze(-1)  # (N, N)

        return logits


import numpy as np
from torch_geometric.data import Data
from collections import defaultdict

class NodeLatentVAEGenerator:
    def __init__(self, train_dataset, model):
        self.train_dataset = train_dataset
        self.model = model

    def sample_graph_from_posterior(self, g):
        self.model.eval()
        with torch.no_grad():
            x = g.x
            edge_index = g.edge_index
            print("MUTAG:")
            print(g)
            print(g.edge_index)

            mu, logstd = self.model.encoder(x, edge_index)
            z = self.model.reparameterize(mu, logstd)

            logits = self.model.decode(z)
            prob_adj = torch.sigmoid(logits)

            prob_adj.fill_diagonal_(0)

            adj = (prob_adj > EDGE_PROBABILITY_CUTOFF).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()
            edge_index_new = adj.nonzero(as_tuple=False).t().contiguous()

            print("TSOUPO:")
            print(Data(edge_index=edge_index_new, num_nodes=g.num_nodes))
            print(prob_adj.mean(), prob_adj.min(), prob_adj.max())

            return Data(edge_index=edge_index_new, num_nodes=g.num_nodes)


    def forward(self, num_samples=1):
        graphs = []
        for _ in range(num_samples):
            idx = torch.randint(0, len(self.train_dataset), (1,)).item()
            g = self.train_dataset[idx]
            graphs.append(self.sample_graph_from_posterior(g))
        return graphs



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

def train_vae(train_dataset, model, epochs=200):
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for data in loader:
            opt.zero_grad()

            x, edge_index, batch = data.x, data.edge_index, data.batch

            mu, logstd = model.encoder(x, edge_index)
            z = model.reparameterize(mu, logstd)

            # -----------------------------
            # INNER PRODUCT DECODER
            # -----------------------------
            #logits = z @ z.t()
            logits = model.decode(z)

            # -----------------------------
            # BUILD ADJACENCY (CORRECT)
            # -----------------------------
            adj = torch.zeros_like(logits)

            for g_id in batch.unique():
                node_mask = (batch == g_id)

                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                edges = edge_index[:, edge_mask]

                adj[edges[0], edges[1]] = 1.0

            # symmetrize (undirected graphs)
            adj = adj + adj.t()
            adj.fill_diagonal_(0)

            # -----------------------------
            # MASK (only same graph pairs)
            # -----------------------------
            mask = batch.unsqueeze(0) == batch.unsqueeze(1)

            # -----------------------------
            # LOSS
            # -----------------------------
            pos = adj[mask].sum()
            neg = mask.sum() - pos
            pos_weight = neg / (pos + 1e-6)

            recon_loss = F.binary_cross_entropy_with_logits(
                logits[mask],
                adj[mask],
                pos_weight=pos_weight
            )

            # -----------------------------
            # KL (stable version)
            # -----------------------------
            kl = -0.5 * torch.mean(
                torch.sum(
                    1 + 2 * logstd - mu**2 - torch.exp(2 * logstd),
                    dim=1
                )
            )

            beta = min(0.1, epoch / 100)

            loss = recon_loss + beta * kl

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")

def sample(num_samples, model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")

    model = NodeVAE(
        in_channels=dataset.num_features,
        hidden_dim=32,
        latent_dim=16
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()

    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    generator = NodeLatentVAEGenerator(train_dataset, model)

    graphs = generator.forward(num_samples)

    output_path = os.path.join("mini3", "generated_graphs_vae.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)

    print(f"Generated {len(graphs)} graphs and saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vae_node_latent.pt", help="Path to the model weights file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    subparsers.add_parser("train")

    # Sample command
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("num_samples", type=int)

    args = parser.parse_args()

    if args.command == "train":
        train(args.model)

    elif args.command == "sample":
        sample(args.num_samples, args.model)


if __name__ == "__main__":
    main()