import LinearRegression

import torch
import matplotlib.pyplot as plt
import numpy as np

X = torch.rand(100, 1) * 10
y = X + 3*torch.rand(100, 1)

x_train = X[:int(0.8*len(X))]
x_test = X[int(0.8*len(X)):]
y_train = y[:int(0.8*len(y))]
y_test = y[int(0.8*len(y)):]

torch.manual_seed(1)
model = LinearRegression.LR(input_size=1, output_size=1)

model.train(learning_rate= 0.01, epochs = 100, x = x_train, y = y_train)

y_predic, e = model.predic(x_test, y_test)

plt.scatter(x_test, y_test)
plt.plot(x_test.numpy(), y_predic.detach().numpy(), 'r')
plt.show()