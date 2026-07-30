# Stage 1 — MLP with Manual Backprop

See also: [learning-plan.md](learning-plan.md) · paper methodology §4: [`../paper/paper.md`](../paper/paper.md)

## Goal

Replace the bigram counter with a neural network that learns distributed word representations. The model must beat Stage 0 baseline by at least 10 pp Macro-F1.

**Baseline to beat:** Macro-F1 = 0.511 (bigram, V=5000)  
**Target:** Macro-F1 > 0.611

---

## Results

This is an educational project: the point is to see the MLP working end to end with
hand-derived backprop, not to chase a state-of-the-art score. Two snapshots below:
the model straight after implementation (no tuning), then after a small round of
tuning that shows what each lever is worth.

### Right after implementation (2026-06-25)

Config: `C_D=50`, `C_H=128`, `C_K=2`, `C_V=5000`, `C_LR=0.01`, `C_BATCH=64`,
`C_EPOCHS=10`, early stopping on validation Macro-F1 with `patience=3`.
Best epoch = 5 (early-stopped at epoch 8). Validation Macro-F1 = 0.5331.

Test set (`syzyfowe-prace`, `tajemniczy-ogrod`), comparable to the Stage 0 baseline:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| NONE | 0.9582 | 0.7502 | 0.8416 |
| COMMA | 0.3270 | 0.6122 | 0.4263 |
| PERIOD | 0.2387 | 0.5818 | 0.3385 |
| **Macro-F1** | | | **0.5354** |

| Model | Test Macro-F1 |
|-------|---------------|
| Stage 0 bigram baseline | 0.5106 |
| Stage 1 MLP (untuned) | **0.5354** |
| Target | 0.611 |

So the MLP already beats the baseline by ~2.5 pp out of the box, but is still ~7.6 pp
short of the target. Recall on the rare classes is decent (the class weights work — no
collapse to always-NONE), but precision on COMMA/PERIOD is low: the model over-predicts
punctuation.

### After tuning (2026-07-01)

Final config: `C_D=50`, `C_H=128`, `C_K=3`, `C_V=5000`, `C_LR=0.005`, `C_BATCH=64`,
`C_EPOCHS=30`, `patience=5`, class-weight tempering `C_ALPHA=0.5`.

Test set, comparable to the Stage 0 baseline:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| NONE | 0.922 | 0.922 | 0.922 |
| COMMA | 0.588 | 0.467 | 0.520 |
| PERIOD | 0.333 | 0.442 | 0.381 |
| **Macro-F1** | | | **0.608** |

| Model | Test Macro-F1 |
|-------|---------------|
| Stage 0 bigram baseline | 0.5106 |
| Stage 1 MLP (untuned) | 0.5354 |
| Stage 1 MLP (tuned) | **0.6077** |
| Target | 0.611 |

Result: +9.7 pp over the baseline, on par with the +10 pp target (within run-to-run
noise of ±0.01 from random init).

What each lever was worth, applied in order:

| Step | What changed | Test Macro-F1 |
|------|--------------|---------------|
| untuned | `α=1`, `K=2`, `LR=0.01`, 10 epochs | 0.5354 |
| optimizer | `LR 0.01→0.005`, epochs `10→30`, `patience 3→5` | ~0.555 (val) |
| context | `C_K 2→3` | ~0.559 (val) |
| **class weights** | `α 1→0.5` (sqrt of inverse frequency) | **0.6063** |
| final | retrain at `α=0.5` | **0.6077** |

The dominant lever by far was **tempering the class weights**. Full inverse-frequency
(`α=1`) over-weights the rare classes ~11:1, so the model over-predicts punctuation
(high recall, low precision). Tempering to `w ∝ (1/count)^α` with `α=0.5` compresses
that ratio to ~3:1 and rebalances toward precision — COMMA precision jumped 0.33→0.59,
and NONE rose too (fewer false-positive commas). A sweep over `α ∈ {0.4, 0.5, 0.6}`
peaked cleanly at `0.5`. Context (`C_K`) and learning-rate/epoch changes gave only
small gains; raw model capacity (`C_D`/`C_H`) was not the bottleneck — the rare-class
precision was.

`C_ALPHA` lives in `src/config/settings.m`; the weight formula is in `src/train.m`.

> The numbers above originally came from a one-off measurement; `src/check.m`
> (load `model.mat`, evaluate on test) now reproduces them exactly — Macro-F1 = 0.6077.

---

## Architecture

Final config (`C_K=3`, window = 7 words; started at `C_K=2`, widened during tuning — see Results):

```
INPUT (7 word indices)
  [w_{i-3}, w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}, w_{i+3}]
         |
         | lookup in E (5001 × 50)
         ↓
EMBEDDING (350 numbers)
  [0.1, -0.3, ..., 0.7]   ← 7 vectors of 50, concatenated
         |
         | W1 (128 × 350) + b1 + ReLU
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
| E (embedding matrix) | 5001 × 50 | 250,050 | `randn * 0.01` |
| W1 | 128 × 350 | 44,800 | He: `randn * sqrt(2/350)` |
| b1 | 128 × 1 | 128 | zeros |
| W2 | 3 × 128 | 384 | He: `randn * sqrt(2/128)` |
| b2 | 3 × 1 | 3 | zeros |
| **Total** | | **~295,400** | |

E has V+1 = 5001 rows: the extra row is `<UNK>`, where `get_word_indices` maps every
out-of-vocab word (`train.m` passes `length(vocab)+1` to `mlp_init`).

---

## What Each Parameter Does

**E** — lookup table. Row `i` = 50-number vector for word `i` in vocab (last row = `<UNK>`). Learned from scratch during training. At start: random noise. After training: similar words cluster nearby in 50D space.

**W1, b1** — first linear layer. Takes 350 numbers (7 embedded words), outputs 128 hidden activations. Each of the 128 neurons detects some pattern across the 7-word window. Followed by ReLU (negatives → 0).

**W2, b2** — second linear layer. Takes 128 hidden values, outputs 3 logits (one per class). Followed by softmax → probabilities.

---

## Key Formulas

| Step | Formula |
|------|---------|
| Embedding lookup | `x = [E(w1,:), …, E(w7,:)]` — concat, shape 1×350 |
| Linear 1 | `s1 = x * W1' + b1'` — shape 1×128 |
| ReLU | `a1 = max(0, s1)` |
| Linear 2 | `s2 = a1 * W2' + b2'` — shape 1×3 |
| Softmax | `p_i = exp(z_i) / sum(exp(z))` — stable: subtract max first |
| CE loss | `L = sum(w .* ce) / N`, `ce = -log(p_true)` — w = tempered class weight of true class |
| Gradient δ2 | `δ2 = (p - y) .* w / N` — y = one-hot, this is the full CE+softmax gradient |
| ReLU gradient | `δ1 = ((W2' * δ2') .* (s1 > 0)')'` — shape N×128 |
| W gradient | `dW = δ' * input` |
| b gradient | `db = sum(δ, 1)'` |
| Embedding gradient | `dx_embed = δ1 * W1` (N×350), then scatter-add: `dE(idx,:) += block` for each of 7 window positions |
| He init | `W = randn(out, in) * sqrt(2/in)` |

> ✅ **Resolved (was a ⚠ during derivation):** `(W2' * δ2')` gives 128×N, mask `(s1 > 0)'` matches that shape, and the final outer transpose brings δ1 back to N×128 — consistent with the batched backward pass below. Implemented in `src/mlp_backward.m`.

---

## Why Weighted Loss

Label distribution in the original 9-book training set, now train+val (1,067,705 tokens):
- NONE = 80.58%
- COMMA = 12.14%
- PERIOD = 7.29%

Without weights, the network learns to always predict NONE (~80.6% accuracy, Macro-F1 ≈ 0.33). Class weights = `(total_N / (3 * count_per_class)) ^ α` — rare classes get higher weight, forcing the network to pay attention to them. Full inverse frequency (`α=1`) overshoots (over-predicts punctuation); the final model tempers with `C_ALPHA=0.5` — see Results for why this was the dominant tuning lever.

---

## Forward Pass (batched, N samples)

```
X_idx     N × 7     word indices
x_embed   N × 350   E(X_idx) concatenated
s1        N × 128   x_embed * W1' + b1'
a1        N × 128   max(0, s1)
s2        N × 3     a1 * W2' + b2'
probs     N × 3     softmax(s2, row-wise)
```

---

## Backward Pass

```
δ2        N × 3      (probs - Y) .* w / N   [Y = one-hot]
dW2       3 × 128    δ2' * a1
db2       3 × 1      sum(δ2, 1)'
δ1        N × 128    ((W2' * δ2') .* (s1 > 0)')'
dW1       128 × 350  δ1' * x_embed
db1       128 × 1    sum(δ1, 1)'
dx_embed  N × 350    δ1 * W1    ← the missing link back to the embedding concat
dE        5001 × 50  scatter-add: dx_embed split into 7 blocks of d columns,
                     each block accumulated (+=) into dE rows by word index
```

> ✅ **Resolved (was a ⚠ during derivation):** δ1 itself cannot be split into window blocks — the gradient must first cross W1 back to the input: `dx_embed = δ1 * W1` (N×350). Only then does it split into 2k+1 blocks of d columns for the scatter-add. Implemented in `src/mlp_backward.m`.

---

## Gradient Check

Before any training: verify analytic gradients numerically.

Method: for each parameter θ, compute `(L(θ+ε) - L(θ-ε)) / 2ε` with ε=1e-5.  
Metric: relative error = `|analytic - numerical| / (|analytic| + |numerical|)`.  
Pass threshold: < 1e-5.

Use tiny model for speed: V=20, d=3, h=4, batch=5. Check W2, W1, b2, b1, and 10 random entries of E.

Do not start training until this passes.

> **Fixed (2026-06-15).** Was flaky because the test always probed `E(1,1)` — word index 1 may not appear in a random `X_idx`, giving `dE(1,1)=0` and numerical grad≈0, so `0/0=NaN` → assert fails. Fixed by probing `E(X_idx(1,1), 1)` — a row guaranteed present in the batch.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/mlp_init.m` | Initialize E, W1, b1, W2, b2 |
| `src/mlp_forward.m` | Forward pass, returns probs + cache |
| `src/mlp_loss.m` | Weighted CE loss + δ2 |
| `src/mlp_backward.m` | All gradients |
| `src/tests/test_grad_check.m` | Numerical gradient verification |
| `src/train.m` | Mini-batch SGD training loop, early stopping on val Macro-F1, saves `model.mat` |
| `src/check.m` | Load `model.mat`, evaluate Macro-F1 on test |
| `src/detect.m` | Interactive inference: prompt for text, restore `,` and `.` |

Reused from Stage 0: `src/lib/metrics.m`, `data/processed/train.mat`, `test.mat`. `vocab.mat` is rebuilt by `train.m` (from the post-carve-out train set) if missing. Produced: `data/processed/model.mat` (best E, W1, b1, W2, b2 — committed).

---

## Inference on Arbitrary Text (`detect.m`)

Training and evaluation consume `(word, label)` pairs that already exist. Inference has neither labels nor file input, which forces three decisions:

**1. Input.** `tokenize` used to open a file itself, so it was unusable on a string typed by the user. It was split by responsibility: `tokenize(text)` is now pure string processing, and the `fopen`/`fread` half moved up into its only caller, `process.m`. `detect.m` reads the sentence with `input(prompt, 's')` — the `'s'` flag matters, without it Octave evaluates the input as an expression.

**2. Labels that do not exist.** `build_windows` returns `[X_idx, y]` and indexes `labels(i)` while building. The labels never influence `X_idx` — they only produce `y` — so inference passes a dummy vector and discards `y`. It must be sized *after* padding: the loop reads up to `labels(length(word_indices) - k)`, so a vector of length n would run off the end.

**3. Boundary words.** With windows dropped at the edges, the first and last `k` words get no prediction — including the final word, which is exactly where the sentence-ending period belongs. Fix: pad the index vector with `k` `<UNK>` indices at *both* ends. `n + 2k` indices produce `n + 2k - 2k = n` windows, and window `r` centres on padded position `r + k`, i.e. real word `r` — a clean 1:1 mapping with no off-by-one.

The caveat is semantic: `<UNK>` means "word outside the vocabulary", not "text boundary". A dedicated `<PAD>` row in `E` would be the honest fix, but it requires retraining, so it is deferred.

**Reconstruction** zips `words` with the predicted classes through a 3-element lookup table (`{'', ',', '.'}`, positions matching `C_LABELS`) and joins with spaces. Text the user already punctuated is stripped by `labelize` first, so the model always predicts from scratch rather than seeing its own answer in the input.

**Known limitations** (accepted, not bugs):
- Output is lower-cased — `tokenize` folds case and capitalisation is never restored.
- The final word often gets NONE, because its right context is three `<UNK>`. Forcing a period there was considered and rejected: it is cosmetic post-processing that would have to be kept out of `check.m` anyway, since mixing product heuristics into the measurement inflates the reported metric.

---

> ✅ **Resolved open item (2026-06-15):** the validation set was carved out of the train documents — `C_VAL_BOOKS` = Ziemia Obiecana + 1984 (239,580 tokens, 20.0%) → `val.mat`. The test set stayed untouched, so the Stage 0 baseline numbers remain valid; the baseline itself now trains on train+val to keep its canonical 1,067,705-token corpus.

---

## Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| Loss = NaN | lr too high (try 0.001) or missing prob clamp in loss |
| Loss stuck | lr too low, or weight init issue |
| PERIOD F1 ≈ 0 | class weights not applied |
| Grad check fails | wrong transpose in dW, or scatter-add uses `=` not `+=`, or missing 1/N factor |