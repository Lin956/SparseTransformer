import torch
import torch.nn as nn
from Layers import MultiHeadAttention, CrossAttention
import torch.nn.functional as F
import math


class Encoder(nn.Module):
    def __init__(self, en_num_layers, embed_size, heads, map_dim):
        super(Encoder, self).__init__()
        self.encoderlayer = EncoderLayer(embed_size, heads, map_dim)
        self.en_num_layers = en_num_layers

    def forward(self, input):
        for i in range(self.en_num_layers):
            out = self.encoderlayer(input)
            input = out
        return out


# input: [C, T, embed_size]
# ouput: [C, T, embed_size]
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, map_dim):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_size, heads)
        self.fc1 = nn.Linear(embed_size, map_dim)
        self.fc2 = nn.Linear(map_dim, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, input):
        x = self.attn(input)
        x = self.norm1(x + input)
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.norm2(x + out)
        return out


# x_en: [C, T, embed_size]
# y_de: [C, out_T_dim, embed_size]
class Decoder(nn.Module):
    def __init__(self, dec_num_layers, embed_size, heads, T, out_T_dim, map_dim):
        super(Decoder, self).__init__()
        self.dec_num_layers = dec_num_layers
        self.decoderlayer = DecoderLayer(embed_size, heads, T, out_T_dim, map_dim)

    def forward(self, x_en, y_de):
        for i in range(self.dec_num_layers):
            out = self.decoderlayer(x_en, y_de)
            y_de = out
        return out


# out: [C, out_T_dim, embed_size]
class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, T, out_T_dim, map_dim):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_size, heads)
        self.crossattn = CrossAttention(embed_size, heads, T, out_T_dim)
        self.fc1 = nn.Linear(embed_size, map_dim)
        self.fc2 = nn.Linear(map_dim, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, x_en, y_de):
        y_de = self.norm1(self.attn(y_de) + y_de)
        crossout = self.norm2(self.crossattn(x_en, y_de) + y_de)
        out = self.fc2(F.relu(self.fc1(crossout)))
        return self.norm3(out + crossout)


"""
x = torch.rand((1, 20, 512))
y = torch.rand((1, 1, 512))
model = Decoder(6, 512, 8, 20, 1, 2048)
out = model(x, y)
print(out.shape)
"""


# input: [C, T, embed_size]
# output: [C, T, embed_size]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


"""
x = torch.rand((1, 3, 1))
model = PositionalEncoding(512)
y = model(x)
print(y.shape)
"""


class discriminator(nn.Module):
    def __init__(self, embed_size, out_T_dim):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(1, embed_size)
        self.fc2 = nn.Linear(embed_size, 2048)
        self.pe = PositionalEncoding(embed_size)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, out_T_dim)

    def forward(self, input):
        input.unsqueeze_(0)  # [1, C, T]
        x = input.permute(1, 2, 0)  # [C, T, 1]
        x_pe = self.pe(x)
        x = F.relu(self.fc1(x)) + x_pe  # [C, T, embed_size]

        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        out = F.leaky_relu(self.fc4(x))
        out.squeeze_(2)

        return out

