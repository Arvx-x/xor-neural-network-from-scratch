import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pytorch_xor.model import XORNet

# --------------------
# Dataset
# --------------------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--lr",     type=float, default=2.0)
parser.add_argument("--epochs", type=int,   default=10000)
args = parser.parse_args()

lr     = args.lr
epochs = args.epochs

# Model, Loss, Optimizer
model     = XORNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training Loop
loss_history = []
print("Training started...")
print("-" * 50)

for epoch in range(epochs):
    logits = model(X)
    loss   = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        loss_history.append(loss.item())
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

# Evaluation
print("\nFinal Results After Training:")
print("-" * 50)
print(f"{'Input':<14} {'Target':<10} {'Prediction':<12} {'Rounded':<10}")
print("-" * 50)

with torch.no_grad():
    logits      = model(X)
    predictions = torch.sigmoid(logits)

for i in range(len(X)):
    raw     = predictions[i][0].item()
    rounded = int(round(raw))
    tgt     = int(y[i][0].item())
    status  = "✅" if rounded == tgt else "❌"
    print(f"{str(X[i].tolist()):<14} {tgt:<10} {raw:<12.4f} {rounded:<10} {status}")

print("-" * 50)

#plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, 'b-', linewidth=0.5, label='PyTorch BCEWithLogitsLoss')
plt.xlabel('Iteration (x1000)')
plt.ylabel('Loss')
plt.title('PyTorch Implementation (BCE)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.savefig("plots/pytorch_mse_loss.png")
print("\nPlot saved → plots/pytorch_mse_loss.png")
