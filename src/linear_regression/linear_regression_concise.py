import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.net = nn.LazyLinear(1)
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # Generate synthetic data: y = 2 * x + 3 + noise
    X = torch.randn(100, 1)
    y = 2 * X + 3 + torch.randn(100, 1) * 0.1

    model = LinearRegression(lr=0.1)
    model.configure_optimizers()

    for epoch in range(20):
        y_hat = model(X)
        loss = model.loss(y_hat, y)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")
