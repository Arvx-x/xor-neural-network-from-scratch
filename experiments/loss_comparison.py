import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pytorch_xor.model import XORNet

# Dataset
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int,   default=50000)
parser.add_argument("--lr_mse", type=float, default=0.01)
parser.add_argument("--lr_bce", type=float, default=0.01)
args = parser.parse_args()

epochs = args.epochs

# Training function
def run_training(loss_fn_name, lr, epochs):
    model     = XORNet()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if loss_fn_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    print(f"\n{'─'*50}")
    print(f"  Training with {loss_fn_name.upper()}  |  lr={lr}  epochs={epochs}")
    print(f"{'─'*50}")

    history = []
    for epoch in range(epochs):
        logits = model(X)

        if loss_fn_name == "mse":
            loss = criterion(torch.sigmoid(logits), y)  # sigmoid before MSE
        else:
            loss = criterion(logits, y)                 # raw logits for BCEWithLogitsLoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            history.append(loss.item())
            print(f"Epoch {epoch:5d} | {loss_fn_name.upper()} Loss: {loss.item():.6f}")

    return history

# Run both
mse_history = run_training("mse", lr=args.lr_mse, epochs=epochs)
bce_history = run_training("bce", lr=args.lr_bce, epochs=epochs)

# Plot
x_ticks = [i * 1000 for i in range(len(mse_history))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"PyTorch XOR — MSE vs BCE Loss Comparison  (epochs={epochs})", fontsize=14, fontweight="bold")

axes[0].plot(x_ticks, mse_history, color="#4C72B0", linewidth=1.5, label=f"MSE (lr={args.lr_mse})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Mean Squared Error (MSE)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_ticks, bce_history, color="#C44E52", linewidth=1.5, label=f"BCE (lr={args.lr_bce})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Binary Cross-Entropy (BCE)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("plots/loss_comparison.png")
plt.close(fig)
print("\nComparison plot saved → plots/loss_comparison.png")
