import torch
import matplotlib.pyplot as plt
from Transfomers import SparseTransformer
import torch.nn as nn
import torch.optim as optim
import numpy as np
from discriminator import Discriminator


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.1)
    data = np.sin(x) + 5 * np.cos(x) + 5*x - 3 * (x**3)
    # print(data.shape)

    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)

    data = torch.tensor(data, dtype=torch.float32)
    data.unsqueeze_(0)

    """
    plt.plot(x, data)
    plt.xlim([-5, 5])
    plt.show()
    """

    en_num_layers = 1
    dec_num_layers = 1
    embed_size = 512
    heads = 8
    T = 20
    out_T_dim = 1
    map_dim = 2048

    epochs = 10
    lr = 0.000001

    D = Discriminator(embed_size, out_T_dim)

    model = SparseTransformer(en_num_layers, dec_num_layers, embed_size, heads, T, out_T_dim, map_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    G_losses = []
    for epoch in range(epochs):
        for i in range(100 - 21):
            x, target_real = data[:, i:i+20], data[:, i+20:i+21]

            # train D
            D_real = D(target_real)

            D_real_loss = torch.log(D_real)

            pred_fake = model(x, target_real).detach()
            D_fake_loss = torch.log(1 - D(pred_fake))

            loss_all = -torch.mean((D_real + D_real_loss), dim=1)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # train SparseTransformer
            target_pred = model(x, target_real)

            G_loss = criterion(target_real, target_pred) + torch.mean(torch.log(1 - D(target_pred)), dim=1)
            G_losses.append(G_loss)

            optimizer.zero_grad()
            G_loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("EPOCH:{},NUM:{}, d_loss:{}, g_loss:{}".format(epoch, i, loss_all.item(), G_loss.item()))


plt.plot(G_losses, label='G_loss')
plt.title("epochs {}, heads {}, lr {}".format(epochs, heads, lr))
plt.legend()
plt.show()

model.eval()

preds = []
for i in range(100 - 21):
    x, y = data[:, i:i+20], data[:, i+20:i+21]

    pred = model(x, y)
    preds.append(pred.reshape(-1))
print(preds)

plt.plot(preds, label='predictions')
plt.plot(data[:, 21:].reshape(-1), label='ground truth')
plt.title("epochs {}, heads {}, lr {}".format(epochs, heads, lr))
plt.legend()
plt.show()

plt.plot(preds, label='predictions')
plt.legend()
plt.show()

plt.plot(data[:, 21:].reshape(-1), label='ground truth')
plt.legend()
plt.show()
