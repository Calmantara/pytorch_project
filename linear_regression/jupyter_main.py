# %%

# %%
# dot product matrix
# a = torch.tensor([1, 2, 3, 4, 5, 6]).view(2, 3)
# b = torch.tensor([2, 3, 4, 2, 3, 4]).view(3, 2)
# torch.matmul(a, b)
# %%
# derivative 
# x = torch.tensor(1.0, requires_grad=True)
# z = torch.tensor(5.0, requires_grad=True)
# y = 10*x**2 + 10*z*2
# y.backward()
# print(x.grad, z.grad)
# %%
# linear regression
# w = torch.tensor(-1.0, requires_grad=True)
# b = torch.tensor(5.0, requires_grad=True)
# def forward(x):
#     y = w*x + b
#     return y
# x = torch.tensor([[2.0, 5.0]])
# print(forward(x))

# %% Data Set
import LinearRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

X = torch.rand(100, 1) * 10
Y = X + 3*torch.rand(100, 1)

torch.manual_seed(1)
model = LinearRegression.LR(input_size=1, output_size=1)
w, b = model.get_param()

# %%
def plot_model(x, y):
    w1, b1 = model.get_param()
    x1 = np.array([0, 10])
    y1 = w1 * x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(x, y)
    plt.show()

plot_model(X, Y)
# %%
