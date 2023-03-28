import numpy as np

# f = w *x
# find w = 2

X = np.array([1,2,3,4], dtype= np.float32)
y = np.array([2,4,6,8], dtype= np.float32)

w = 0.0

# model predict
def forward(w,X):
    return w*X

# use lose = MSE
def loss(y,y_pre):
    return ((y-y_pre)**2).mean()

def greadient(X,y,y_pred):
    return np.dot(2*X,y_pred-y).mean()

epoch = 10
lr = 0.01
for i in range(epoch):
    # y_pred
    y_pred = forward(w,X)

    # loss function
    l = loss(y,y_pred)

    # dl/dw
    dw = greadient(X,y,y_pred)

    w -= lr*dw

    print(f'epoch{i+1}: w = {w}, loss = {l}')