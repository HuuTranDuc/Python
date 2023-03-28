import torch

x = torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True)
y = x**2
z = y.mean()

z.backward()   #dz/dx

print(x.grad)  # Output: tensor([0.666, 1.3333, 2.00000])

# Xóa đạo hàm thì dùng
# x.grad.zero_()
