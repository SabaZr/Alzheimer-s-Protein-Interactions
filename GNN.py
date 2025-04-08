import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        """
        Initializes an advanced 3-layer Graph Convolutional Network with batch normalization and dropout.
        
        Args:
            input_dim (int): Dimensionality of input features.
            hidden_dim (int): Number of hidden units for intermediate layers.
            output_dim (int): Dimensionality of output embeddings.
            dropout (float): Dropout probability for regularization.
        """
        super(GNN, self).__init__()  # Changed from AdvancedGNN to GNN
        # First GCN layer: input -> hidden
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        # Second GCN layer: hidden -> hidden (deeper representation)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        # Third GCN layer: hidden -> output
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the Advanced GCN.
        
        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph edge index.
            
        Returns:
            Tensor: Output node embeddings.
        """
        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Third layer
        x = self.conv3(x, edge_index)
        return x

def convert_nx_to_pyg(G_ppi, node_features):
    """
    Converts a NetworkX graph to PyTorch Geometric Data format.
    """
    nodes_in_graph = list(G_ppi.nodes())
    node_map = {node: i for i, node in enumerate(nodes_in_graph)}
    edges = [[node_map[u], node_map[v]] for u, v in G_ppi.edges() if u in node_map and v in node_map]
    edge_index = np.array(edges).T
    print("Edge index shape (before tensor conversion):", edge_index.shape)
    
    if edge_index.size == 0:
        raise ValueError("No edges found after conversion; please check your G_ppi graph.")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    print("Edge index shape (after tensor conversion):", edge_index.shape)
    
    x = torch.tensor(node_features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def train_gnn(graph_data, hidden_dim=64, output_dim=None, epochs=200, lr=0.01):
    """
    Trains a GCN and extracts node embeddings using unsupervised MSE loss.
    The output dimension is set to match the input dimension if not specified.
    """
    if output_dim is None:
        output_dim = graph_data.x.shape[1]

    model = GNN(input_dim=graph_data.x.shape[1], hidden_dim=hidden_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = F.mse_loss(out, graph_data.x)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    embeddings = model(graph_data.x, graph_data.edge_index).detach().numpy()
    return embeddings
