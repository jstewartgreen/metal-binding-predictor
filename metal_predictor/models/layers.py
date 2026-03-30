import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MetalMPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='sum')
        self.message_mlp = MLP(2 * node_dim + edge_dim, hidden_dim)
        self.update_mlp  = MLP(node_dim + hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.update_mlp(torch.cat([x, msg], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
