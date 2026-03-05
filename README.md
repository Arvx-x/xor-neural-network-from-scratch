# XOR Neural Network — From Scratch

> A pedagogical deep learning project implementing an XOR-solving neural network in raw NumPy and PyTorch, with experiments exploring loss functions and learning rate dynamics.

---

## Table of Contents
1. [Problem Description](#problem-description)
2. [Motivation](#motivation)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Backpropagation — Core Math](#backpropagation--core-math)
6. [Implementation](#implementation)
7. [Experiments & Results](#experiments--results)
   - [NumPy MSE Training](#1-numpy-mse-training)
   - [PyTorch BCE Training](#2-pytorch-bce-training)
   - [Loss Function Comparison: MSE vs BCE](#3-loss-function-comparison-mse-vs-bce)
   - [Learning Rate Effect](#4-learning-rate-effect)
8. [Key Insights](#key-insights)
9. [Running the Experiments](#running-the-experiments)
10. [Requirements](#requirements)
11. [Appendix: Full Mathematical Derivation](#appendix-full-mathematical-derivation)

---

## Problem Description

The **XOR (exclusive OR)** function is a classic non-linearly separable binary classification problem. Given two binary inputs, the output is 1 only when the inputs differ:

| Input A | Input B | XOR Output |
|:-------:|:-------:|:----------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line can separate the positive class `{(0,1), (1,0)}` from the negative class `{(0,0), (1,1)}` in 2D space. This makes XOR the canonical benchmark for demonstrating that a neural network with at least one hidden layer can learn non-linear decision boundaries — something a simple perceptron cannot.

---

## Motivation

This project serves two purposes:

1. **Pedagogical** — Build a complete neural network from first principles using only NumPy, with every forward pass, loss computation, and gradient manually coded. This makes the internals of deep learning transparent, as opposed to using a framework that abstracts them away.

2. **Comparative** — Implement the same model in PyTorch to contrast framework-level abstractions (autograd, `nn.Module`, optimizers) against manual implementations. Run controlled experiments on loss functions (MSE vs BCE) and learning rate dynamics to demonstrate their theoretical properties empirically.

---

## Project Structure

```
xor-nn-from-scratch/
│
├── numpy_xor/
│   ├── model.py              # XORModel class: forward(), backward(), train()
├── pytorch_xor/
│   └── model.py              # XORNet (nn.Module): 2→2→1, raw logit output
│
├── experiments/
│   ├── numpy_training.py     # NumPy model, MSE → plots/numpy_mse_loss.png
│   ├── pytorch_training.py   # PyTorch model, BCE → plots/pytorch_mse_loss.png
│   ├── loss_comparison.py    # PyTorch MSE vs BCE → plots/loss_comparison.png
│   └── learning_rate.py      # LR effect (lr=100 vs lr=0.001) → plots/lr_effect.png
│
├── plots/
│   ├── numpy_mse_loss.png
│   ├── pytorch_mse_loss.png
│   ├── loss_comparison.png
│   └── lr_effect.png
│
└── README.md
```

---

## Model Architecture

Both implementations share the same architecture: a fully connected feedforward network with one hidden layer.

```
Input Layer       Hidden Layer        Output Layer
  (2 units)  →   (2 units, σ)   →    (1 unit, σ)
```

- **Input:** 2 features (the two XOR bits)
- **Hidden layer:** 2 neurons with sigmoid activation
- **Output layer:** 1 neuron — sigmoid activation (NumPy) or raw logit (PyTorch with `BCEWithLogitsLoss`)
- **Parameters:** W1 (2×2), b1 (1×2), W2 (2×1), b2 (1×1) = **9 trainable parameters total**

The use of sigmoid activations introduces non-linearity, allowing the network to carve out a non-linear decision boundary in input space — the minimum requirement to solve XOR.

---

## Backpropagation — Core Math

Training uses **gradient descent** with **backpropagation** (chain rule applied layer-by-layer in reverse).

### Forward Pass

```math
Z^{[1]} = X W^{[1]} + b^{[1]}
```
```math
A^{[1]} = \sigma(Z^{[1]})
```
```math
Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}
```
```math
\hat{y} = \sigma(Z^{[2]})
```

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ and its derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### Loss Functions

**Mean Squared Error (MSE):**
```math
\mathcal{L}_{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2
```

**Binary Cross-Entropy (BCE):**
```math
\mathcal{L}_{BCE} = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
```

### Output Layer Gradients

The critical difference between the two loss functions is visible in the output-layer gradient.

**With MSE:**
```math
\frac{\partial \mathcal{L}}{\partial Z^{[2]}} = \frac{2}{m}(\hat{y} - y) \cdot \sigma'(Z^{[2]}) = \frac{2}{m}(\hat{y} - y) \cdot \hat{y}(1-\hat{y})
```

**With BCE:**
```math
\frac{\partial \mathcal{L}}{\partial Z^{[2]}} = \frac{1}{m}(\hat{y} - y)
```

The sigmoid derivative $\hat{y}(1-\hat{y})$ cancels analytically in the BCE case, yielding a clean error signal. In the MSE case, this term multiplies the gradient and approaches zero whenever $\hat{y}$ saturates toward 0 or 1—the root cause of the **vanishing gradient problem** in MSE-trained sigmoid networks.

### Weight Updates (Gradient Descent)

```math
W^{[2]} \leftarrow W^{[2]} - \alpha \cdot (A^{[1]})^\top \delta^{[2]}
```
```math
W^{[1]} \leftarrow W^{[1]} - \alpha \cdot X^\top \delta^{[1]}
```

where $\alpha$ is the learning rate and $\delta^{[\ell]}$ denotes the layer-$\ell$ error term.

---

## Implementation

### NumPy (`numpy_xor/model.py`)

The `XORModel` class holds weights as plain NumPy arrays and exposes three methods:

```python
model = XORModel(seed=45)

# Single step
y_hat = model.forward(X)           # forward pass
model.backward(y, lr=4.0, loss_fn="mse")  # backprop + weight update

# Or full training loop
history = model.train(X, y, lr=4.0, epochs=30000, loss_fn="mse")  # or "bce"
```

Every matrix operation, sigmoid call, and gradient computation is written explicitly. There is no automatic differentiation.

### PyTorch (`pytorch_xor/model.py`)

```python
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)   # outputs raw logits

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        return self.output(x)
```

Training uses `nn.BCEWithLogitsLoss()` (numerically stable, fuses sigmoid + BCE) and `optim.SGD`. Backpropagation is handled entirely by PyTorch's autograd engine — no gradients are manually coded.

---

## Experiments & Results

All experiments use the same 4-sample XOR dataset and SGD optimizer. Seeds are fixed for reproducibility.

### 1. NumPy MSE Training

**Setup:** `lr=4.0`, `epochs=30000`, MSE loss, manual backprop.

![NumPy MSE Loss](plots/numpy_mse_loss.png)

**Result:** The network converges to near-zero MSE loss within the first ~2000 iterations and achieves perfect classification on all four XOR inputs (all predictions correctly rounded to 0 or 1). This confirms the manual implementation of forward pass and backpropagation is mathematically correct.

---

### 2. PyTorch BCE Training

**Setup:** `lr=2.0`, `epochs=10000`, `BCEWithLogitsLoss`, PyTorch autograd.

![PyTorch BCE Loss](plots/pytorch_mse_loss.png)

**Result:** The PyTorch model achieves similarly rapid convergence, with BCE loss dropping sharply within the first ~1000–2000 epochs. The model correctly classifies all XOR inputs. The slightly different curve shape compared to MSE reflects the different loss scale (BCE is unbounded above; MSE is bounded at 0.25 for binary problems).

---

### 3. Loss Function Comparison: MSE vs BCE

**Setup:** Same PyTorch `XORNet`, `lr=0.01` for both, `epochs=50000`. Learning rate intentionally kept low to expose gradient quality differences.

![Loss Comparison](plots/loss_comparison.png)

**Result and Analysis:**

This experiment directly demonstrates the **vanishing gradient problem** in MSE-trained sigmoid networks.

- **MSE (left):** Loss drops from ~0.259 to ~0.250 in the first few hundred epochs, then becomes completely flat for the remaining ~49,500 epochs. A final loss of `0.2499` is approximately the theoretical maximum MSE for a binary problem where $\hat{y} \approx 0.5$ — indicating the model has learned nothing meaningful and is producing near-uniform predictions.

- **BCE (right):** Loss plateaus near `0.693` (i.e., $\ln 2$, the value when $\hat{y} = 0.5$) for the first ~25,000 epochs, then begins a sustained, accelerating descent, reaching `0.369` by epoch 50,000 and clearly still converging.

**Cause:** The MSE output gradient $\frac{\partial \mathcal{L}}{\partial Z^{[2]}} = \frac{2}{m}(\hat{y}-y)\cdot\hat{y}(1-\hat{y})$ is damped by the sigmoid derivative term $\hat{y}(1-\hat{y}) \approx 0.25$ at initialisation, and approaches zero as sigmoid saturates. At `lr=0.01`, this dampening is sufficient to stall training entirely. The BCE gradient $\frac{1}{m}(\hat{y}-y)$ has no such dampening — it maintains a gradient signal proportional purely to the prediction error, allowing sustained learning at the same learning rate.

> **Under identical hyperparameters and architecture, BCE converges while MSE stalls — a direct consequence of vanishing gradients induced by MSE's sigmoid derivative term.**

---

### 4. Learning Rate Effect

**Setup:** PyTorch `XORNet`, `BCEWithLogitsLoss`, `epochs=500`. Two extreme learning rates compared: `lr=100` and `lr=0.001`.

![Learning Rate Effect](plots/lr_effect.png)

**Result:**

- **Too high (`lr=100`, left):** The loss spikes to ~20 within the first few epochs and then oscillates violently between ~4 and ~20 for all 500 epochs — never converging. Each gradient step massively overshoots the loss minimum, sending weights to the opposite extreme and producing an unstable feedback loop. The model never learns XOR.

- **Too low (`lr=0.001`, right):** The loss curve is a near-perfect straight line, decreasing linearly from `0.709` to `0.705` over 500 epochs. While stable, the change is so small that meaningful convergence would require tens of thousands more epochs at this rate.

**Insight:** The learning rate controls the trade-off between stability and speed. Too high — the optimizer overshoots; too low — it takes impractically long to converge. In practice, a well-tuned learning rate (e.g., `0.1`–`2.0` for this problem with SGD) produces rapid, stable convergence.

---

## Key Insights

| # | Insight |
|---|---------|
| 1 | A single hidden layer with sigmoid activations is sufficient to solve the non-linearly separable XOR problem |
| 2 | The NumPy and PyTorch implementations are mathematically equivalent; the framework handles bookkeeping, not the math |
| 3 | **BCE is theoretically and empirically superior to MSE for binary classification with sigmoid outputs** due to the vanishing gradient problem in MSE |
| 4 | `BCEWithLogitsLoss` is preferred over manually applying sigmoid then BCE — it uses the log-sum-exp trick for numerical stability, avoiding NaN when logits become extreme |
| 5 | Learning rate is the most sensitive hyperparameter: too high causes oscillatory divergence; too low causes impractical convergence speed |
| 6 | For XOR, most of the convergence happens within the first ~10–15% of total training epochs regardless of loss function (at sufficient lr), suggesting early stopping could be highly effective |

---

## Running the Experiments

All scripts must be run from the **project root** using `-m` (no `sys.path` hacks needed):

```bash
# NumPy model, MSE
python -m experiments.numpy_training

# PyTorch model, BCE
python -m experiments.pytorch_training

# Loss comparison (MSE vs BCE)
python -m experiments.loss_comparison

# Learning rate effect
python -m experiments.learning_rate
```

All scripts accept CLI arguments with sensible defaults:

```bash
python -m experiments.numpy_training --lr 4.0 --epochs 30000
python -m experiments.pytorch_training --lr 2.0 --epochs 10000
python -m experiments.loss_comparison --epochs 50000 --lr_mse 0.01 --lr_bce 0.01
```

Plots are saved automatically to `plots/`.

---

## Requirements

```
numpy
matplotlib
torch
```

```bash
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Appendix: Full Mathematical Derivation

<details>
<summary><strong>📐 Complete Backpropagation Derivation — NumPy Implementation</strong></summary>

### A. Notation

_Coming soon._

### B. Forward Pass — Full Derivation

_Coming soon._

### C. Loss Function Gradients

_Coming soon._

### D. Output Layer Backpropagation

_Coming soon._

### E. Hidden Layer Backpropagation

_Coming soon._

### F. Weight Update Equations

_Coming soon._

</details>
