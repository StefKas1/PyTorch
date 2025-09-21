import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
        n = y_hat.shape[0]
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
    # 1. Load MNIST Dataset
    # ---------------------------
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten 28x28 to 784
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # ---------------------------
    # 2. Initialize model & optimizer
    # ---------------------------
    input_dim = 28 * 28
    num_classes = 10
    model = SoftmaxClassifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = SGD(model.parameters, lr=0.1)

    # ---------------------------
    # 3. Training Loop
    # ---------------------------
    epochs = 5
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for batch_X, batch_y in train_loader:
            y_hat = model.forward(batch_X)
            loss = model.loss(batch_y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
            preds = y_hat.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    # ---------------------------
    # 4. Test Accuracy
    # ---------------------------
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model.forward(X)
            preds = y_hat.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)

    print(f"\nTest Accuracy: {correct / total:.4f}")
