import argparse
import numpy as np
import matplotlib.pyplot as plt

from numpy_xor.model import XORModel, LOSS_FUNCTIONS

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--lr",     type=float, default=2.0)
parser.add_argument("--epochs", type=int,   default=30000)
args = parser.parse_args()

lr     = args.lr
epochs = args.epochs

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Init model
model = XORModel(seed=45)
compute_loss = LOSS_FUNCTIONS["mse"]

loss_history = []
print("Training started...")
print("-" * 50)

# Training Loop
for epoch in range(epochs):
    y_hat = model.forward(X)
    loss  = compute_loss(y, y_hat)
    model.backward(y, lr=lr, loss_fn="mse")

    if epoch % 2000 == 0:
        loss_history.append(loss)
        print(f"Iteration {epoch:5d} | Loss: {loss:.6f}")

# Final Predictions
print("\nFinal Results After Training:")
print("-" * 50)
print(f"{'Input':<12} {'Target':<10} {'Prediction':<12} {'Rounded':<10}")
print("-" * 50)

y_hat = model.forward(X)
for i in range(len(X)):
    rounded = round(y_hat[i][0])
    status  = "✅" if rounded == y[i][0] else "❌"
    print(f"{str(X[i]):<12} {y[i][0]:<10} {y_hat[i][0]:<12.4f} {rounded:<10} {status}")

print("-" * 50)
print("The network learned XOR from random weights!")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', linewidth=0.5, label='NumPy (from scratch)')
plt.xlabel('Iteration (x2000)')
plt.ylabel('Loss')
plt.title('NumPy Implementation (MSE)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig("plots/numpy_mse_loss.png")
print("\nPlot saved → plots/numpy_mse_loss.png")
