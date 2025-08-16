import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, lr, loss_fn, weight_decay=0):
        super().__init__()
        self.net = nn.LazyLinear(1)
        self.lr = lr
        self.loss_fn = loss_fn
        self.optimizer = None
        self.weight_decay = weight_decay  # 0, without L2 regularization

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def train(model, X, y, epochs=20):
    for epoch in range(epochs):
        y_hat = model(X)
        loss = model.loss(y_hat, y)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    # Generate synthetic data: y = 2 * x + 3 + noise
    X = torch.randn(100, 1)
    y = 2 * X + 3 + torch.randn(100, 1) * 0.1

    # With MSELoss
    loss_fn = nn.MSELoss()
    model = LinearRegression(lr=0.1, loss_fn=loss_fn)
    model.configure_optimizers()
    train(model, X, y, epochs=20)

    # With HuberLoss
    loss_fn = nn.HuberLoss()
    model = LinearRegression(lr=0.1, loss_fn=loss_fn)
    model.configure_optimizers()
    train(model, X, y, epochs=20)
