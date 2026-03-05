import numpy as np

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Loss (MSE)
def compute_loss(y, y_hat):
    return np.mean((y - y_hat) ** 2)


# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])


# Initialize Weights
np.random.seed(45)

W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))


# Training Parameters
lr = 4.0
epochs = 30000

loss_history = []
print("Training started...")
print("-" * 50)


# Training Loop
for epoch in range(epochs):

    # ---- Forward Pass ----
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    y_hat = sigmoid(Z2)

    # ---- Loss ----
    loss = compute_loss(y, y_hat)

    # ---- Backprop ----
    m = len(y)

    # Output layer gradients
    dZ2 = (2/m) * (y_hat - y) * sigmoid_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ---- Update Weights ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Print progress
    if epoch % 2000 == 0:
        loss = compute_loss(y, y_hat)
        loss_history.append(loss)
        print(f"Iteration {epoch:5d} | Loss: {loss:.6f}")

# Final Predictions

print("Final Results After Training:")
print("-" * 50)
print(f"{'Input':<12} {'Target':<10} {'Prediction':<12} {'Rounded':<10}")
print("-" * 50)

for i in range(len(X)):
    rounded = round(y_hat[i][0])
    status = "✅" if rounded == y[i][0] else "❌"
    print(f"{str(X[i]):<12} {y[i][0]:<10} {y_hat[i][0]:<12.4f} {rounded:<10} {status}")

print("-" * 50)
print("The network learned XOR from random weights!")