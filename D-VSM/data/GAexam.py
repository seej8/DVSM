import warnings

import torch

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv

class GATv1(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(GATv1, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_dim, heads=3)
        self.conv2 = GATConv(3 * hidden_dim, hidden_dim, heads=3)
        self.conv3 = SAGEConv(3 * hidden_dim,3 * hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_dim, out_channels))

    def forward(self, x, edge_index, adj=None):
        x, edge_index = x, edge_index
        x = self.conv1(x, edge_index)
        print("conv1 : ", x.shape)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        print("conv1 drop1 : ", x.shape)
        x = self.conv2(x, edge_index)
        print("conv2 : ", x.shape)
        x = F.dropout(F.relu(x), p=0.5, training=self.training)
        print("drop2 : ", x.shape)
        x = self.conv3(x,edge_index)
        print("conv3 :", x.shape)
        x = self.mlp(x)
        print("mlp : ", x.shape)
        return F.sigmoid(x)


a = torch.ones((5, 5))
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
model = GATv1(5, 64, 32)
b = model(a, edge_index)
