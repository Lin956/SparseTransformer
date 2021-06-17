import pandas as pd
import torch
import numpy as np

def get_data():
    df = pd.read_table("./electricity.txt", sep=',', header=None)

    data = np.array(df)
    data = torch.tensor(data, dtype=torch.float32)
    data = torch.transpose(data, 0, 1)

    """
    print(data.shape)  [C, T]
    """
    return data

"""
data = get_data()
print(data.shape)
print(data)
"""
