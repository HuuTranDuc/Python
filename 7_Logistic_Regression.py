import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#0 prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_sample, n_feauter = X.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

#Scale
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

input_size = n_feauter

#1 Design model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression,self).__init__()
        #define layer
        self.lin = nn.Linear(input_dim,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.lin(x))
        return (y_pred)

model = LogisticRegression(input_size)

#2 loss and optimizer
criterion = nn.BCELoss()

lrate = 0.01
optim = torch.optim.SGD(params=model.parameters(), lr = lrate)

#3 training loop
iter = 100
for epoch in range(iter):
    
    #forward pass and predict
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    #backward
    loss.backward()

    #update
    optim.step()
    optim.zero_grad()

    if epoch %10 == 0:
        [w,b] = model.parameters()
        print(f'epcho:{epoch}, l: {loss.item()}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round() #làm tròn y
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy:{acc}')

