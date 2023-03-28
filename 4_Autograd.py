import torch

# f = w *x
# find w = 2

X = torch.tensor([1,2,3,4], dtype= torch.float32)
y = torch.tensor([2,4,6,8], dtype= torch.float32)

w = torch.tensor(0.0, dtype =torch.float32, requires_grad = True)

# model predict
def forward(w,X):
    return w*X

# use lose = MSE
def loss(y,y_pre):
    return ((y-y_pre)**2).mean()


epoch = 100
lr = 0.01
for i in range(epoch):
    # y_pred
    y_pred = forward(w,X)

    # loss function
    l = loss(y,y_pred)

    # dl/dw
    l.backward()  

    # Off grad
    with torch.no_grad():
        w -= lr*w.grad

    # delete grad for new loop
    w.grad.zero_()
    if i %10 ==0:
        print(f'epoch{i+1}: w = {w}, loss = {l}')