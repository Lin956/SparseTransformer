import torch
import torch.nn as nn
from modules import Encoder, Decoder, PositionalEncoding
import torch.nn.functional as F


# input: [C, T]
# output: [C, out_T_dim]
class SparseTransformer(nn.Module):
    def __init__(self, en_num_layers, dec_num_layers, embed_size, heads, T, out_T_dim, map_dim):
        super(SparseTransformer, self).__init__()
        self.pe = PositionalEncoding(embed_size)
        self.encoder = Encoder(en_num_layers, embed_size, heads, map_dim)
        self.decoder = Decoder(dec_num_layers, embed_size, heads, T, out_T_dim, map_dim)
        self.fc1 = nn.Linear(1, embed_size)
        self.fc2 = nn.Linear(1, embed_size)
        self.fc3 = nn.Linear(embed_size, 1)

    def forward(self, x, y):
        """
        x is input: [C, T]
        y is target: [C, out_T_dim]
        """
        x.unsqueeze_(0)  # [1, C, T]
        x = x.permute(1, 2, 0)  # [C, T, 1]
        x_pe = self.pe(x)
        x_en = F.relu(self.fc1(x)) + x_pe  # [C, T, embed_size]
        # print(x_en.shape)

        y.unsqueeze_(2)  # [C, out_T_dim, 1]
        y_pe = self.pe(y)
        y_de = F.relu(self.fc2(y)) + y_pe
        # print(y_de.shape)

        en_out = self.encoder(x_en)
        dec_out = self.decoder(en_out, y_de) # [C, out_T_dim, embed_size]

        target_out = self.fc3(dec_out)
        return target_out.squeeze_(2)


"""
x = torch.rand((3, 20))
y = torch.rand((3, 1))
model = SparseTransformer(4, 4, 512, 8, 20, 1, 2048)
out = model(x, y)
print(out.shape)
"""
