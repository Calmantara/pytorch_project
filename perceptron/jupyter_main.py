# %%
import torch

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
# %%
num_pts = 100
centre = [[-0.5, 0.5], [0.5, -0.5]]
X, y = datasets.make_blobs(n_samples=num_pts, random_state=123, centers=centre, cluster_std=0.4)
x_data = torch.Tensor(X)
y_data = torch.Tensor(y)
# %%
def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
# plt.show()
# %%
class Classification(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._input_size = input_size
        self._classification = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        pred = torch.sigmoid(self._classification(x))
        return pred

    def predic(self, x):
        predic = self.forward(x)
        return 1 if predic >= 0.5 else 0

    def get_parameter(self):
        w, b = self._classification.parameters()
        w0 = []
        for i in w.view(self._input_size):
            w0.append(i.item())
        return w0, b[0].item()

    def train(self, learning_rate:float, epochs:int, x, y):
        epochs = epochs
        # losses = [] 

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self._classification.parameters(), lr = learning_rate)
        
        for i in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            print("epoch:", i, "loss:", loss.item())
            # losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#make initial model
torch.manual_seed(2)
model = Classification(input_size=2, output_size=1)
# w, b = model.get_parameter()

# %%
def plot_fit(title):
    plt.title = title
    w, b = model.get_parameter()
    x1 = np.array([-2, 2])
    x2 = (w[0]*x1 + b) / (-w[1])
    plt.plot(x1, x2, 'r')
    scatter_plot()
    plt.plot()

# %%
plot_fit('initial model')

# %%
model.train(learning_rate=0.1, epochs=1000, x=x_data, y=y_data)
plot_fit('trained model')

# %%
