import LinearRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

X = torch.rand(100, 1) * 10
Y = X + 3*torch.rand(100, 1)

torch.manual_seed(1)
model = LinearRegression.LR(input_size=1, output_size=1)

model.train(learning_rate= 1, epochs = 100, x = X, y = Y)

model.plot_model(plt, np, X, Y)