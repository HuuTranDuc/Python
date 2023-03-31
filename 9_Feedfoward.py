# MNIST
# DataLoader, Transformation
# Mutilayer Neural Net, activation function
# Loss and Optimizer
# Traininng loop
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper paramaters
input_size = 784 #24x24   MNIST size
hidden_size = 100
num_class = 10

num_epoch = 2
lrate = 0.001
batch_size = 100

# MNIST  dataset
train_dataset = torchvision.datasets.MNIST(root= './', train= True, transform= transform.ToTensor(), download= True)
test_dataset = torchvision.datasets.MNIST(root= './', train= False, transform= transform.ToTensor())

# data loader
train_loader = DataLoader(dataset= train_dataset, batch_size = batch_size, shuffle= True)
test_loader = DataLoader(dataset=test_dataset, batch_size = batch_size, shuffle= False)

# examples = iter(train_loader)
# samples, lables = next(examples)
# print(samples.shape, lables.shape)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, num_class)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_class)
model.to(device)

# loss and opimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lrate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, lables) in enumerate(train_loader):
        # batch, channel, w,h
        # 100,1,28,28
        # 100,784
        # linear data
        images = (images.view(-1, 28*28)).to(device)
        lables = lables.to(device)

        # foward
        outputs = model(images)
        loss = criterion(outputs,lables)

        # backward and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if (i+1)%100 == 0:
        #     print(f'epoch{epoch+1}/{num_epoch}, step{i+1}/{n_total_steps}, loss: {loss}')

#phase test
with torch.no_grad():
    n_correct = 0
    n_sample = 0
    for i, (images, lables) in enumerate(test_loader):
        
        images = images.reshape(-1,28*28).to(device)
        lables = lables.to(device)

        output = model(images)

        _, prediction = torch.max(output, 1)
        n_sample += lables.shape[0]
        n_correct += (prediction == lables).sum().item()

    acc = n_correct/n_sample*100
    print(f'accuracy: {acc}')

