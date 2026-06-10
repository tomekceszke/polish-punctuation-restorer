# Stage 1 — MLP with Manual Backprop

See also: [learning-plan.md](learning-plan.md) · paper methodology §4: [`../paper/paper.md`](../paper/paper.md)

## Goal

Replace the bigram counter with a neural network that learns distributed word representations. The model must beat Stage 0 baseline by at least 10 pp Macro-F1.

**Baseline to beat:** Macro-F1 = 0.511 (bigram, V=5000)  
**Target:** Macro-F1 > 0.611

---

## Architecture

```
INPUT (5 word indices)
  [w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}]
         |
         | lookup in E (5000 × 50)
         ↓
EMBEDDING (250 numbers)
  [0.1, -0.3, ..., 0.7]   ← 5 vectors of 50, concatenated
         |
         | W1 (128 × 250) + b1 + ReLU
         ↓
HIDDEN LAYER (128 numbers)
  [0, 0.4, 0, 1.2, 0, ...]  ← zeros from ReLU
         |
         | W2 (3 × 128) + b2 + softmax
         ↓
OUTPUT (3 probabilities)
  [0.1, 0.7, 0.2]
   NONE  COM  PER
```

---

## Parameters

| Name | Shape | Count | Init |
|------|-------|-------|------|
| E (embedding matrix) | 5000 × 50 | 250,000 | `randn * 0.01` |
| W1 | 128 × 250 | 32,000 | He: `randn * sqrt(2/250)` |
| b1 | 128 × 1 | 128 | zeros |
| W2 | 3 × 128 | 384 | He: `randn * sqrt(2/128)` |
| b2 | 3 × 1 | 3 | zeros |
| **Total** | | **~282,500** | |

---

## What Each Parameter Does

**E** — lookup table. Row `i` = 50-number vector for word `i` in vocab. Learned from scratch during training. At start: random noise. After training: similar words cluster nearby in 50D space.

**W1, b1** — first linear layer. Takes 250 numbers (5 embedded words), outputs 128 hidden activations. Each of the 128 neurons detects some pattern across the 5-word window. Followed by ReLU (negatives → 0).

**W2, b2** — second linear layer. Takes 128 hidden values, outputs 3 logits (one per class). Followed by softmax → probabilities.

---

## Key Formulas

| Step | Formula |
|------|---------|
| Embedding lookup | `x = [E(w1,:), E(w2,:), E(w3,:), E(w4,:), E(w5,:)]` — concat, shape 1×250 |
| Linear 1 | `s1 = x * W1' + b1'` — shape 1×128 |
| ReLU | `a1 = max(0, s1)` |
| Linear 2 | `s2 = a1 * W2' + b2'` — shape 1×3 |
| Softmax | `p_i = exp(z_i) / sum(exp(z))` — stable: subtract max first |
| CE loss | `L = -mean(w_k * log(p_k))` — w_k = class weight for true class |
| Gradient δ2 | `δ2 = (p - y) * w_k / N` — y = one-hot, this is the full CE+softmax gradient |
| ReLU gradient | element-wise: `δ1 = (W2' * δ2') .* (s1 > 0)'` |
| W gradient | `dW = δ * input'` |
| b gradient | `db = sum(δ, 1)'` |
| Embedding gradient | scatter-add: `dE(idx,:) += δ_embed` for each of 5 indices |
| He init | `W = randn(out, in) * sqrt(2/in)` |

> ⚠ **Recheck the ReLU gradient row:** what shape does `(W2' * δ2') .* (s1 > 0)'` actually produce, and what shape does the backward pass below declare for δ1? The two disagree — rederive before implementing.

---

## Why Weighted Loss

Label distribution in training set (1,067,705 tokens):
- NONE = 80.58%
- COMMA = 12.14%
- PERIOD = 7.29%

Without weights, the network learns to always predict NONE (~80.6% accuracy, Macro-F1 ≈ 0.33). Class weights = `total_N / (3 * count_per_class)` — rare classes get higher weight, forcing the network to pay attention to them.

---

## Forward Pass (batched, N samples)

```
X_idx     N × 5     word indices
x_embed   N × 250   E(X_idx) concatenated
s1        N × 128   x_embed * W1' + b1'
a1        N × 128   max(0, s1)
s2        N × 3     a1 * W2' + b2'
probs     N × 3     softmax(s2, row-wise)
```

---

## Backward Pass

```
δ2        N × 3     (probs - Y) .* w / N   [Y = one-hot]
dW2       3 × 128   δ2' * a1
db2       3 × 1     sum(δ2, 1)'
δ1        N × 128   (W2' * δ2') .* (s1 > 0)'  [transposed carefully]
dW1       128 × 250 δ1' * x_embed
db1       128 × 1   sum(δ1, 1)'
dE        5000 × 50 scatter-add from δ1 split into 5 blocks of d columns
```

> ⚠ **Recheck the dE row:** δ1 is N×128 — it cannot split into 5 blocks of 50 columns. A step is missing between δ1 and the embedding concat. What carries the gradient from the hidden layer back to the 250-dim input x_embed? Derive it before implementing.

---

## Gradient Check

Before any training: verify analytic gradients numerically.

Method: for each parameter θ, compute `(L(θ+ε) - L(θ-ε)) / 2ε` with ε=1e-5.  
Metric: relative error = `|analytic - numerical| / (|analytic| + |numerical|)`.  
Pass threshold: < 1e-5.

Use tiny model for speed: V=20, d=3, h=4, batch=5. Check W2, W1, b2, b1, and 10 random entries of E.

Do not start training until this passes.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/mlp_init.m` | Initialize E, W1, b1, W2, b2 |
| `src/mlp_forward.m` | Forward pass, returns probs + cache |
| `src/mlp_loss.m` | Weighted CE loss + δ2 |
| `src/mlp_backward.m` | All gradients |
| `src/tests/test_grad_check.m` | Numerical gradient verification |
| `src/train.m` | Mini-batch SGD training loop (WIP) |
| `src/check.m` | (planned) Load weights, evaluate Macro-F1 on test |

Reused from Stage 0: `src/lib/metrics.m`, `data/processed/train.mat`, `test.mat`, `vocab.mat`.

**Open item:** early stopping needs a validation set — none exists yet (current split is 90/10 train/test). Carve it out of the train documents before training; the test set stays untouched so the Stage 0 baseline numbers remain valid.

---

## Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| Loss = NaN | lr too high (try 0.001) or missing prob clamp in loss |
| Loss stuck | lr too low, or weight init issue |
| PERIOD F1 ≈ 0 | class weights not applied |
| Grad check fails | wrong transpose in dW, or scatter-add uses `=` not `+=`, or missing 1/N factor |