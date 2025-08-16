import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Linear Regression Model
# ---------------------------
class LinearRegression:
    def __init__(self, size, l2_lambda=0):
        self.w = torch.normal(mean=0, std=0.01, size=size, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.l2_lambda = l2_lambda  # 0, without L2 regularization

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def _l2_penalty(self):
        return self.l2_lambda * (torch.norm(self.w, p=2) ** 2 / 2)

    def loss(self, y, y_hat):
        squared_error = (y - y_hat) ** 2 / 2
        mse = squared_error.mean()

        return mse + self._l2_penalty()

    @property
    def parameters(self):
        return [self.w, self.b]


# ---------------------------
# Custom SGD Optimizer
# ---------------------------
class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
                # ^^ Above is correct because param.grad contains the
                # partial derivative of the loss function which contains L2 penalty

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


if __name__ == "__main__":
    # ---------------------------
    # 1. Generate Data
    # ---------------------------
    true_w = torch.tensor([[3.0]])
    true_b = 2.0
    X = torch.randn(1000, 1)
    y = X @ true_w + true_b + 0.1 * torch.randn(1000, 1)

    # ---------------------------
    # 2. DataLoader
    # ---------------------------
    batch_size = 32
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # 3. Initialize model & optimizer
    # ---------------------------
    model = LinearRegression(size=(1, 1))
    optimizer = SGD(model.parameters, lr=0.05)

    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            y_hat = model.forward(batch_X)
            loss = model.loss(batch_y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # ---------------------------
    # 5. Print learned weights
    # ---------------------------
    print(f"\nLearned w: {model.w.data}, b: {model.b.item():.4f}")
