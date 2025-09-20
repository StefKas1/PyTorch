import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Softmax Classification Model
# ---------------------------
class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes):
        # Parameters: weight matrix and bias
        self.W = torch.normal(
            mean=0, std=0.01, size=(input_dim, num_classes), requires_grad=True
        )
        self.b = torch.zeros(num_classes, requires_grad=True)

    def forward(self, X):
        logits = torch.matmul(X, self.W) + self.b
        return torch.softmax(logits, dim=1)

    def loss(self, y, y_hat):
        # y: (batch,) long tensor of class indices
        # y_hat: (batch, num_classes)
        n = y_hat.shape[0]
        # Negative log likelihood
        log_likelihood = -torch.log(y_hat[range(n), y])
        return log_likelihood.mean()

    @property
    def parameters(self):
        return [self.W, self.b]


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

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


if __name__ == "__main__":
    # ---------------------------
    # 1. Generate Classification Data
    # ---------------------------
    num_samples = 1000
    num_features = 2
    num_classes = 3

    # Simple synthetic dataset (3 clusters)
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))

    # ---------------------------
    # 2. DataLoader
    # ---------------------------
    batch_size = 32
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------
    # 3. Initialize model & optimizer
    # ---------------------------
    model = SoftmaxClassifier(input_dim=num_features, num_classes=num_classes)
    optimizer = SGD(model.parameters, lr=0.1)

    # ---------------------------
    # 4. Training Loop
    # ---------------------------
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        for batch_X, batch_y in dataloader:
            y_hat = model.forward(batch_X)
            loss = model.loss(batch_y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = y_hat.argmax(dim=1)
            correct += (preds == batch_y).sum().item()

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    # ---------------------------
    # 5. Print learned weights
    # ---------------------------
    print(f"\nLearned W:\n{model.W.data}\nLearned b: {model.b.data}")
