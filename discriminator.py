import torch
import torch.nn as nn
import torch.nn.functional as F


# input:[C, out_T_dim]
# output: [C, out_T_dim]
class Discriminator(nn.Module):
    def __init__(self, embed_size, out_T_dim):
        super(Discriminator, self).__init__()
        """
        三层全连接层
        """
        self.fc1 = nn.Linear(1, embed_size)
        self.fc2 = nn.Linear(embed_size, 2048)
        self.fc3 = nn.Linear(2048, 1)
        self.l_relu1 = nn.LeakyReLU(0.2)
        self.l_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        x_b = x.unsqueeze(2)

        x_b = self.l_relu1(self.fc1(x_b))
        x_b = self.l_relu2(self.fc2(x_b))
        x_b = self.fc3(x_b)

        out = x_b.squeeze(0)

        return out


"""
x = torch.randn((12, 2))
model = Discriminator(512, 2)
out = model(x)
print(x.shape)
"""
