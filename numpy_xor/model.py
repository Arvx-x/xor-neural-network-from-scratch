import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def mse_loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def bce_loss(y, y_hat):
    y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

LOSS_FUNCTIONS = {"mse": mse_loss, "bce": bce_loss}


class XORModel:

    def __init__(self, seed=45):
        np.random.seed(seed)
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.y_hat = sigmoid(self.Z2)
        self.X = X
        return self.y_hat

    def backward(self, y, lr=4.0, loss_fn="mse"):
        m = len(y)
        if loss_fn == "mse":
            dZ2 = (2 / m) * (self.y_hat - y) * sigmoid_derivative(self.Z2)
        elif loss_fn == "bce":
            dZ2 = (self.y_hat - y) / m

        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr=4.0, epochs=30000, loss_fn="mse"):  # noqa: E501
        self.lr = lr
        loss_history = []
        compute_loss = LOSS_FUNCTIONS[loss_fn]

        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss  = compute_loss(y, y_hat)
            self.backward(y, loss_fn=loss_fn)

            if epoch % 2000 == 0:
                loss_history.append(loss)
                print(f"Epoch {epoch:6d} | {loss_fn.upper()} Loss: {loss:.6f}")

        return loss_history
