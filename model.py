import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, dropout):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=dropout, training=self.training)
        return self.conv2(x, data.edge_index)
