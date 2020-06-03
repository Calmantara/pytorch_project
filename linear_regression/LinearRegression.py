import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__ (self, input_size: int, output_size: int):
        super().__init__()
        self._linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        predic = self._linear(x)
        return predic
    
    def get_param(self):
        [w, b] = self._linear.parameters()
        return w[0][0].item(), b[0].item()
    
    def plot_model(self, plt, np, x, y):
        w1, b1 = self.get_param()
        x1 = np.array([0, 10])
        y1 = w1 * x1 + b1
        plt.plot(x1, y1, 'r')
        plt.scatter(x, y)
        plt.show()

    def train(self, learning_rate:float, epochs:int, x, y):
        epochs = epochs
        # losses = [] 

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self._linear.parameters(), lr = learning_rate)
        
        for i in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            print("epoch:", i, "loss:", loss.item())
            # losses.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()