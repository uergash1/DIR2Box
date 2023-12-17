import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Attention, self).__init__()
        self.dim = embedding_dim
        self.W_key = nn.Linear(embedding_dim, hidden_dim, bias=False)
        #self.W_val = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.query = nn.Parameter(torch.randn(hidden_dim, 1))

    def forward(self, x):
        batch_size, num_document, dim = x.shape

        key = self.W_key(x)
        query = self.query.unsqueeze(0).repeat(batch_size, 1, 1)

        attention = torch.softmax(torch.bmm(key, query) / (self.dim ** 0.5), 1)
        aggregation = torch.sum(attention * x, 1)

        return aggregation
