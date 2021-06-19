import torch
import matplotlib.pyplot as plt
from Transfomers import SparseTransformer
import torch.nn as nn
import torch.optim as optim
import numpy as np


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.1)
    data = np.sin(x) + 5 * np.cos(x) + 5*x
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
    heads = 4
    T = 20
    out_T_dim = 1
    map_dim = 2048

    epochs = 50
    lr = 0.000001

    model = SparseTransformer(en_num_layers, dec_num_layers, embed_size, heads, T, out_T_dim, map_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    losses = []
    target_preds = []
    for epoch in range(epochs):
        for i in range(100 - 21):
            x, target_real = data[:, i:i+20], data[:, i+20:i+21]

            target_pred = model(x, target_real)
            target_preds.append(target_pred)

            loss = criterion(target_real, target_pred)
            losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("Epoch {} MAELoss {:}: {:}".format(epoch, i, loss.item()))


plt.plot(target_preds)
plt.show()
plt.plot(losses, label='Mean Squared Loss')
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
plt.legend()
plt.show()

plt.plot(preds, label='predictions')
plt.legend()
plt.show()

plt.plot(data[:, 21:].reshape(-1), label='ground truth')
plt.legend()
plt.show()


