import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj


class ErdosRenyi:
    """
    Baseline model that samples graphs from an Erdos-Renyi distribution
    with parameters estimated from the training data.
    """

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        self.node_counts = [graph.num_nodes for graph in self.train_dataset]

    def sample_num_nodes(self, num_samples=1):
        """
        Sample the number of nodes from the empirical distribution of node counts
        in the training data.

        Args:
            num_samples: Number of samples to draw (default: 1)

        Returns:
            int or array: Sampled node count(s)
        """
        return np.random.choice(self.node_counts, size=num_samples)

    def link_probability(self, N: np.ndarray):
        """
        Compute the link probability r as the graph density (number of edges divided by total possible
        number of edges) computed from the training graphs with N nodes.

        Args:
            N: Array of node counts to filter by
            dataset: The dataset to filter from

        Returns:
            float: The link probability r
        """
        idx = np.argwhere(np.isin(np.array(self.node_counts), N))
        total_edges = defaultdict(int)
        total_possible_edges = defaultdict(int)
        for i in idx.flatten():
            graph = self.train_dataset[i]
            n = graph.num_nodes
            total_edges[n] += graph.num_edges // 2
            total_possible_edges[n] += n * (n - 1) // 2

        r = {}
        for n in set(N):
            if total_possible_edges[n] > 0:
                r[n] = total_edges[n] / total_possible_edges[n]
            else:
                print(f"Warning: no training graphs with {n} nodes; using r=0.0 (empty graphs will be generated)", file=sys.stderr)
                r[n] = 0.0

        return r

    def sample_graph(self, N: np.ndarray, r: dict) -> list[Data]:
        """
        Sample Erdos-Renyi graphs with variable node counts.

        Args:
            N: Array of number of nodes for each graph
            r: Dictionary mapping node count to link probability

        Returns:
            List of torch_geometric.data.Data objects, one per graph
        """
        result = []
        for n in N:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if np.random.rand() < r[n]:
                        edges.append([i, j])
                        edges.append([j, i])

            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # Create PyG Data object for each graph
            graph_data = Data(edge_index=edge_index, num_nodes=n)
            result.append(graph_data)

        return result

    def forward(self, num_samples=1):
        """
        Sample graphs from the Erdos-Renyi distribution.

        Args:
            num_samples: Number of graphs to sample (default: 1)

        Returns:
            List of torch_geometric.data.Data objects
        """
        N = self.sample_num_nodes(num_samples=num_samples)
        r = self.link_probability(N)
        graphs = self.sample_graph(N, r)
        adjacency_matrices = [
            self.object_to_adjacency_matrix(graph) for graph in graphs
        ]
        return graphs, adjacency_matrices

    def object_to_adjacency_matrix(self, graph: Data):
        """
        Convert a PyG Data object to an adjacency matrix.

        Args:
            graph: A torch_geometric.data.Data object containing edge_index and num_nodes

        Returns:
            A: Adjacency matrix of shape [num_nodes, num_nodes]
        """
        return (
            to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)
            .squeeze(0)
            .numpy()
        )


class CustomDataset(InMemoryDataset):
    def __init__(self, listOfDataObjects):
        super().__init__()
        self.length = len(listOfDataObjects)
        self.data, self.slices = self.collate(listOfDataObjects)

    def __len__(self):
        return self.length


if __name__ == "__main__":
    args = sys.argv[1:]
    assert len(args) == 1, "Usage: python erdos_renyi.py <num_samples>"
    num_samples = int(args[0])
    dataset = TUDataset(root="./data/", name="MUTAG")
    # Split into training and validation
    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, (100, 44, 44), generator=rng
    )

    er = ErdosRenyi(train_dataset)
    graphs, adj_matrices = er.forward(num_samples=num_samples)
    dataset = CustomDataset(graphs)
    torch.save(graphs, "generated_graphs_erdos_renyi.pt")

    ## Usage:
    # Load the saved graphs (weights_only=False needed for PyG Data objects)
    saved_graphs = torch.load("generated_graphs_erdos_renyi.pt", weights_only=False)

    # Create CustomDataset from saved graphs
    custom_dataset = CustomDataset(saved_graphs)

    print(f"CustomDataset loaded!")
    print(f"  Number of graphs: {len(custom_dataset)}")
    print(f"  Type: {type(custom_dataset)}")

    # Get the first graph
    first_graph = custom_dataset[0]
    print(f"\nFirst graph:")
    print(f"  Num nodes: {first_graph.num_nodes}")
    print(f"  Num edges: {first_graph.edge_index.shape[1]}")

    # Convert to adjacency matrix
    first_adj = to_dense_adj(first_graph.edge_index, max_num_nodes=first_graph.num_nodes).squeeze(0).numpy()
    print(f"\nAdjacency matrix of first graph:")
    print(first_adj)

