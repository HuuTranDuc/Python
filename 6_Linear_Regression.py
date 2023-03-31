import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0 prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(Y_numpy.astype(np.float32))
y = y.view(y.shape[0],-1)

n_sample, n_feauter = X.shape
input_size = n_feauter
output_size = 1

#1 Design model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layer
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

#2 loss and optimizer
criterion = nn.MSELoss()

lrate = 0.01
optim = torch.optim.SGD(params=model.parameters(), lr = lrate)

#3 training loop
iter = 100
for epoch in range(iter):
    
    #forward pass and predict
    y_pred = model(X)
    loss = criterion(y_pred, y)

    #backward
    loss.backward()

    #update
    optim.step()
    optim.zero_grad()

    if epoch %10 == 0:
        [w,b] = model.parameters()
        print(f'epcho:{epoch}, w:{w}, l: {loss}')


predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()