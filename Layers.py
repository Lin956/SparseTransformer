import torch
import torch.nn as nn
from entmax.activations import entmax15


# shape of input: [C, T, embed_size]
# shape of output: [C, T, embed_size]
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)

    def forward(self, x):
        C, T, E = x.shape

        x = x.view(C, T, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        # scaled dot-product
        attn = entmax15(torch.matmul(queries, torch.transpose(keys, 2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [C, T, heads, heads]
        # print(attn.shape)
        out = torch.matmul(attn, values)  # [C, T, heads, per_dim]
        # print(out.shape)

        out = out.view(C, T, self.heads*self.per_dim)
        return out


# x_de: [C, T, embed_size]
# y_de: [C, out_T_dim, embed_size]
# out: [C, out_T_dim.,embed_size]
class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads, T, out_T_dim):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.per_dim = embed_size // heads
        self.queries = nn.Linear(self.per_dim, self.per_dim)
        self.keys = nn.Linear(self.per_dim, self.per_dim)
        self.values = nn.Linear(self.per_dim, self.per_dim)
        self.conv1 = nn.Conv2d(T, out_T_dim, 1)

    def forward(self, x_en, y_de):
        C, T, E = x_en.shape
        C_y, T_y, E_y = y_de.shape

        x = x_en.view(C, T, self.heads, self.per_dim)
        x = self.conv1(x)  # [C, out_T_dim, self.per_dim]
        y = y_de.view(C, T_y, self.heads, self.per_dim)

        # compute queries, keys and values
        queries = self.queries(y)
        keys = self.keys(x)
        values = self.values(x)

        # scaled dot-product
        attn = entmax15(torch.matmul(queries, torch.transpose(keys, 2, 3))
                        / (self.embed_size ** (1 / 2)), dim=-1)  # [C, T, heads, heads]
        out = torch.matmul(attn, values)  # [C, T, heads, per_dim]

        out = out.view(C, T_y, self.heads*self.per_dim)
        return out

"""
model = CrossAttention(512, 8, 20, 1)
x = torch.randn(1, 20, 512)
y = torch.randn(1, 1, 512)
out = model(x, y)
print(out.shape)
"""
