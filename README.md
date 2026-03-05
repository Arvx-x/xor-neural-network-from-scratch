# XOR Neural Network — From Scratch

A clean, pedagogical implementation of an XOR-solving neural network, written both in **raw NumPy** and **PyTorch**. The goal is to illustrate how the same model can be built at different levels of abstraction and to compare MSE vs BCE loss dynamics.

---

## Project Structure

```
xor-nn-from-scratch/
│
├── numpy_xor/
│   ├── __init__.py
│   └── model.py          # XORModel class (forward, backward, update)
│
├── pytorch_xor/
│   ├── __init__.py
│   └── model.py          # XORModel (nn.Module)
│
├── experiments/
│   ├── __init__.py
│   ├── numpy_training.py   # NumPy model, MSE → plots/numpy_mse_loss.png
│   ├── pytorch_training.py # PyTorch model, MSE → plots/pytorch_mse_loss.png
│   └── loss_comparison.py  # NumPy MSE vs BCE → plots/loss_comparison.png
│
├── plots/                  # Auto-created when experiments run
│   ├── numpy_mse_loss.png
│   ├── pytorch_mse_loss.png
│   └── loss_comparison.png
│
└── README.md
```

---

## Model Architecture

```
Input (2)  →  Hidden (2, Sigmoid)  →  Output (1, Sigmoid)
```

Both the NumPy and PyTorch implementations use the same 2 → 2 → 1 architecture with sigmoid activations.

---

## Running the Experiments

All scripts must be run from the **project root** (`XOR-NN-from-scratch/`) using `-m` so Python resolves imports correctly — no path hacks needed.

### NumPy — MSE training
```bash
python -m experiments.numpy_training
```

### PyTorch — MSE training
```bash
python -m experiments.pytorch_training
# optional flags:
python -m experiments.pytorch_training --epochs 50000 --lr 0.05
```

### Loss comparison (MSE vs BCE)
```bash
python -m experiments.loss_comparison
# optional flags:
python -m experiments.loss_comparison --epochs 30000 --lr_mse 4.0 --lr_bce 1.0
```

---

## Loss Functions

| Name | Formula | Notes |
|------|---------|-------|
| **MSE** | `mean((y - ŷ)²)` | Default; smooth gradient near 0 |
| **BCE** | `-mean(y·log(ŷ) + (1-y)·log(1-ŷ))` | Standard for binary classification |

---

## numpy_xor/model.py — API

```python
from numpy_xor.model import XORModel, LOSS_FUNCTIONS

model = XORModel(seed=42)
y_hat = model.forward(X)          # forward pass
grads = model.backward(y, loss_fn="mse")  # or "bce"
model.update(grads, lr=4.0)       # gradient descent step
preds = model.predict(X)          # binary (0/1) predictions
```

---

## Requirements

```
numpy
matplotlib
torch
```

Install with:
```bash
pip install numpy matplotlib torch
```
