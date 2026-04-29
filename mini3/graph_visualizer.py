import torch
import argparse
import random
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.utils import to_networkx


from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx


def load_mutag(root="./data/"):
    dataset = TUDataset(root=root, name="MUTAG")
    return dataset

def visualize_mutag(num_graphs=5):
    dataset = load_mutag()

    print(f"Loaded MUTAG: {len(dataset)} graphs")

    sample = dataset[:num_graphs]

    for i, g in enumerate(sample):
        print(f"\nMUTAG Graph {i}")
        print("Nodes:", g.num_nodes)
        print("Edges:", g.edge_index.shape[1])

        G = to_networkx(g, to_undirected=True)

        plt.figure(figsize=(4, 4))
        plt.title(f"MUTAG Graph {i}")

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=300, edge_color="gray", with_labels=False)

        plt.show()

def load_graphs(path):
    data = torch.load(path, weights_only=False)

    # Handle both single graph or list of graphs
    if isinstance(data, list):
        return data
    return [data]


def plot_graph(data, title="Graph"):
    # Convert PyG -> NetworkX
    G = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(4, 4))
    plt.title(title)

    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        font_size=8,
        edge_color="gray"
    )

    plt.show()


def visualize(path, num_graphs=5):
    graphs = load_graphs(path)

    print(f"Loaded {len(graphs)} graphs")

    # sample subset if too many
    sample = random.sample(graphs, min(num_graphs, len(graphs)))

    for i, g in enumerate(sample):
        print(f"\nGraph {i}:")
        print(f"  Nodes: {g.num_nodes}")
        print(f"  Edges: {g.edge_index.shape[1]}")

        plot_graph(g, title=f"Graph {i}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to .pt file with graphs")
    parser.add_argument("--n", type=int, default=5, help="Number of graphs to visualize")

    args = parser.parse_args()

    if args.path == "mutag":
        visualize_mutag(args.n)
    else:
        visualize(args.path, args.n)


if __name__ == "__main__":
    main()