'''
1 Design model(input, output size, forward pass)
2 Construct (loss, optimizer)
3 Training loop
  -forward pass: compute predict
  -backward pass: gradient
  -update weights
'''
# function 
# f = w * x
# find w = 2  

import torch
import torch.nn as nn

# Tensor 2D test
X_test = torch.tensor([8], dtype= torch.float32)

# Tensor 2D train
X = torch.tensor([[1],[2],[3],[4]], dtype= torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype= torch.float32)

n_samples, n_features = X.size()
input_size = n_features
output_size = n_features

# model predict
# model = nn.Linear(input_size, output_size)
class LineaRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LineaRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(X)

model = LineaRegression(input_size,output_size)

# use lose = MSE
loss = nn.MSELoss()
lrate = 0.01
optim = torch.optim.SGD(model.parameters(), lr = lrate)
epoch = 100

for i in range(epoch):

    # delete grad for new loop
    optim.zero_grad()

    # y_pred
    y_pred = model(X)

    # loss function
    l = loss(y,y_pred)

    # dl/dw
    l.backward()
    optim.step()

    if i %10 ==0:
        [w,b] = model.parameters()
        print(f'epoch{i+1}: w = {w.item()}, loss = {l}')
