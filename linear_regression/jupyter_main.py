# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
# %% dot product matrix
a = torch.tensor([1, 2, 3, 4, 5, 6]).view(2, 3)
b = torch.tensor([2, 3, 4, 2, 3, 4]).view(3, 2)
torch.matmul(a, b)
# %% derivative 
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(5.0, requires_grad=True)
y = 10*x**2 + 10*z*2
y.backward()
print(x.grad, z.grad)
# %% linear regression
# w = torch.tensor(-1.0, requires_grad=True)
# b = torch.tensor(5.0, requires_grad=True)
# def forward(x):
#     y = w*x + b
#     return y
# x = torch.tensor([[2.0, 5.0]])
# print(forward(x))
# %% multi variabel
torch.manual_seed(2)
model = torch.nn.Linear(in_features=2, out_features=1)

# %%
print(model.bias, model.weight)

# %%
X = torch.rand(10).view(5,2)
print(X)
print(model(X))
