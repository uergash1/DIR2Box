from attention import Attention
from box_models import *
from torch_geometric.nn import GCNConv, GraphConv, LGConv
import torch.nn.functional as F


class QueryModel(nn.Module):
    def __init__(self, hidden_dim, bert_dim=768):
        super(QueryModel, self).__init__()
        self.layer = nn.Linear(bert_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.layer(x)


class GNN(nn.Module):
    def __init__(self, hidden_dim, edge_index, edge_weight):
        super(GNN, self).__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        return x


class Model(nn.Module):
    def __init__(self, box_type, hidden_dim, edge_index, edge_weight, use_gnn=True, bert_dim=768):
        super(Model, self).__init__()
        # Geometric or Attentive
        self.box_type = box_type
        self.use_gnn = use_gnn

        self.attention = Attention(bert_dim, hidden_dim)

        if self.use_gnn:
            self.gnn = GNN(hidden_dim, edge_index, edge_weight)

        if self.box_type == 'geometric':
            self.box = GeometricBox(hidden_dim)

    def forward(self, x):
        x = self.attention(x)
        if self.use_gnn:
            x = self.gnn(x) + x
        return self.box(x)

