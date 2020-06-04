import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__ (self, input_size: int, output_size: int) -> None:
        super().__init__()
        self._input_size = input_size
        self._linear = nn.Linear(input_size, output_size)

    def forward(self, x) -> list:
        predic = self._linear(x)
        return predic
    
    def predic(self, x, y):
        _error = y - self.forward(x)
        return self.forward(x), "accuracy :" + str(torch.mean(_error).item())

    def get_param(self) -> (list, float):
        [w, b] = self._linear.parameters()
        w0 = []
        for i in w.view(self._input_size):
            w0.append(i.item())
        return w0, b.item()
    
    def plot_model(self, plt, np, x, y) -> None:
        w1, b1 = self.get_param()
        x1 = np.array([0, 10])
        y1 = w1 * x1 + b1
        plt.plot(x1, y1, 'r')
        plt.scatter(x, y)
        plt.show()

    def train(self, learning_rate:float, epochs:int, x, y) -> None:
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