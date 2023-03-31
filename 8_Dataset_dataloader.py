import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#dataset
class Winedataset(Dataset):
    def __init__(self):
        # load data
        file_path = '.\wine.csv'
        data = np.loadtxt(file_path, delimiter=",", dtype= np.float32, skiprows=1)

        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, [0]])
        self.n_sample = data.shape[0]

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = Winedataset()

#dataloader
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
num_epochs = 2
total_sample = len(dataset)
n_iterations = math.ceil(total_sample/4)

for epoch in range(num_epochs):
    for i, (inputs, lables) in enumerate(dataloader):
        if (i+1) % 5 ==0:
            print(f'epoch{epoch}/{num_epochs}, step: {i+1}/{n_iterations}, inputs {inputs.shape}')
