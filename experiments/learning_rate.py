import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

from pytorch_xor.model import XORNet

# Dataset
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

epochs   = 500
lr_high  = 100.0
lr_low   = 0.001


def run(lr, epochs):
    model     = XORNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = []
    for epoch in range(epochs):
        logits = model(X)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = loss.item()
        # capture NaN/Inf as-is so the plot shows the divergence
        history.append(val)

        if epoch % 50 == 0:
            tag = "NaN/Inf ⚠️" if not math.isfinite(val) else f"{val:.6f}"
            print(f"  Epoch {epoch:4d} | Loss: {tag}")

        # stop if fully diverged
        if not math.isfinite(val):
            print(f"  → Diverged at epoch {epoch}. Stopping early.")
            break

    return history


print("=" * 50)
print(f"  HIGH lr = {lr_high}")
print("=" * 50)
history_high = run(lr_high, epochs)

print()
print("=" * 50)
print(f"  LOW lr = {lr_low}")
print("=" * 50)
history_low = run(lr_low, epochs)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Learning Rate Effect on Training  (BCE, epochs={epochs})", fontsize=14, fontweight="bold")

# High lr panel
x_high = list(range(len(history_high)))
axes[0].plot(x_high, history_high, color="#C44E52", linewidth=1.2, label=f"lr = {lr_high}")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title(f"Too High LR ({lr_high}) — Divergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
if any(not math.isfinite(v) for v in history_high):
    axes[0].text(0.5, 0.5, "Diverged (NaN/Inf)", transform=axes[0].transAxes,
                 fontsize=13, color="red", ha="center", va="center",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# Low lr panel
x_low = list(range(len(history_low)))
axes[1].plot(x_low, history_low, color="#4C72B0", linewidth=1.2, label=f"lr = {lr_low}")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("BCE Loss")
axes[1].set_title(f"Too Low LR ({lr_low}) — Barely Learning")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("plots/lr_effect.png")
plt.close(fig)
print("\nPlot saved → plots/lr_effect.png")
