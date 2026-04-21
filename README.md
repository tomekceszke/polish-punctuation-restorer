# Polish Punctuation Restorer

An educational project — building a sequential punctuation classifier from scratch in GNU Octave, deriving all math by hand and implementing backprop manually on matrices. No external ML libraries.

**Task:** given a Polish word, predict the punctuation mark that follows it: none, comma, or period.

**Model type:** sequence labeling / token classification (NLP task: punctuation restoration).

**Stage 1 model:** shallow feedforward neural network — 3-layer MLP, ~280K parameters.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Embedding matrix E | 5000 × 50 | 250,000 |
| W1 | 128 × 250 | 32,000 |
| b1 | 128 | 128 |
| W2 | 3 × 128 | 384 |
| b2 | 3 | 3 |
| **Total** | | **~282,500** |

For reference: GPT-2 small has 117M parameters — roughly 400× more.

---

## Progress

| Stage | Status | Done when |
|-------|--------|-----------|
| **Stage 0 — Preprocessing** | ✅ Done | `preprocess.m` runs, `data.mat` committed |
| **Stage 0 — Bigram baseline** | 🔄 In progress | `baseline_ngram.m` + F1 report in `notes/stage-0-results.md` |
| **Stage 1 — MLP + backprop** | ⬜ Planned | Gradient check passes, macro-F1 beats baseline by ≥10 pp |
| **Stage 2a — Bi-LSTM** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 2b — Mini-Transformer** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 3 — Extended punctuation** | ⬜ Outlook | — |
| **Stage 4 — Multi-task** | ⬜ Outlook | — |
| **Stage 5 — REST deploy** | ⬜ Optional | — |

### Current task

**Stage 0, Step 4** — implement `baseline_ngram.m`:
- For each pair `(word_indices(i), word_indices(i+1))` count `count[label | idx1, idx2]` in `zeros(N+1, N+1, 3)`
- Prediction = argmax with Laplace smoothing
- Evaluate on test set, report Precision / Recall / F1 per class + Macro-F1
- Save results to `notes/stage-0-results.md`

Reference: `notes/stage-0-bigram-baseline.md`

---

## Current State

**Stage 0 — Preprocessing complete.**

- `src/preprocess.m` tokenizes raw `.txt` files, strips non-Polish characters, and emits `(word, label)` pairs.
- Corpus: *Lalka* (Prus) and *Chłopi* (Reymont) from [Wolne Lektury](https://wolnelektury.pl).
- Output: `data/processed/data.mat` — committed for reproducibility.

### Label encoding

| Label | Value | Meaning |
|-------|-------|---------|
| No punctuation | `0` | Plain word |
| Comma | `1` | Word followed by `,` |
| Period | `2` | Word followed by `.` |

---

## Project Structure

```
ppr/
├── data/
│   ├── raw/                 # Source .txt files (Wolne Lektury)
│   └── processed/
│       ├── data.mat         # words (1×N cell) + labels (1×N int) — committed
│       └── vocab.mat        # word_indices integer vector — committed
├── src/
│   ├── preprocess.m         # Entrypoint: raw text → data.mat
│   ├── vocab.m              # Build top-N vocabulary, map words to indices
│   ├── config/
│   │   └── settings.m       # Shared constants (e.g. C_CUT_OFF_WORDS)
│   └── lib/
│       ├── tokenize.m       # Lowercase, strip, split on whitespace
│       └── labelize.m       # Attach labels, strip trailing punctuation
├── notes/
│   ├── learning-plan.md          # Full learning curriculum (EN)
│   ├── stage-0-bigram-baseline.md # Theory + reference for Stage 0 Step 3
│   └── stage-0-results.md        # ← to be created after baseline runs
```

Planned additions (per learning plan):

```
src/
├── baseline_ngram.m         # Stage 0: n-gram frequency baseline
├── mlp_forward.m            # Stage 1: forward pass
├── mlp_backward.m           # Stage 1: manual backprop
├── learn.m                  # Training loop (mini-batch SGD / Adam)
├── check.m                  # Evaluation on test set
├── detect.m                 # Inference on arbitrary text
└── evaluate.m               # Precision / Recall / F1 per class
Theta1.mat, Theta2.mat       # Saved MLP weights
E.mat                        # Embedding matrix
```

---

## Getting Started

**Requirements:** GNU Octave (`brew install octave` on macOS).

```bash
# Run preprocessing (from src/)
cd src
octave-cli preprocess.m
```

Output is written to `../data/processed/data.mat` as `words` and `labels`.

**VS Code:** install the *Octave Debugger* extension. `.vscode/launch.json` is configured to run the current file with `octave-cli`.

To add more source texts, edit the `books` cell array in `src/preprocess.m`:

```matlab
books = {'../data/raw/lalka.txt', '../data/raw/chlopi.txt'};
```

---

## Learning Roadmap

This project follows a 5-stage curriculum, building complexity incrementally.

### Stage 0 — Statistical Baseline *(preprocessing done, baseline in progress)*

N-gram frequency model: for each word pair `(w_i, w_{i+1})`, predict the most common following punctuation using Laplace-smoothed counts.

- Teaches: corpus handling, Polish preprocessing pitfalls, class imbalance (~85% NONE), accuracy vs F1.
- Target: F1 ~0.4–0.55 for PERIOD, ~0.15–0.3 for COMMA.
- Done when: `baseline_ngram.m` runs + F1 report in `notes/stage-0-results.md`.

### Stage 1 — MLP with Hand-Written Backprop *(planned)*

**Architecture:**

```
input:  [w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}]   (5 word indices)
  ↓ embedding lookup  E ∈ R^{V×d}
  ↓ concat → x ∈ R^{5d}
  ↓ W1 ∈ R^{h×5d}, b1,  ReLU
  ↓ W2 ∈ R^{3×h},  b2,  softmax
output: distribution over {NONE, COMMA, PERIOD}
```

Starting hyperparameters: `V=5000, d=50, h=128, k=2, batch=64, lr=0.01`.

Key implementation steps:
1. Build vocabulary with `<UNK>` and `<PAD>`.
2. Forward pass with activation cache.
3. Weighted cross-entropy loss (class weights = inverse frequency).
4. Backward pass — all gradients derived by hand, including embedding scatter-add.
5. Gradient check: numerical vs analytic difference < 1e-6.
6. Xavier/He weight initialisation.
7. SGD with momentum → Adam; observe the difference empirically.
8. Training loop with early stopping; save `Theta1.mat`, `Theta2.mat`, `E.mat`.

Math to derive (in `notes/`): softmax Jacobian, softmax + CE simplification to `p − y`, ReLU gradient, embedding gradient (scatter-add), full chain-rule graph.

Done when: gradient check passes, macro-F1 beats baseline by ≥10 pp.

### Stage 2a — Bi-LSTM *(outlook)*

BPTT, gradient through time, gates. First contact with sequential memory.

### Stage 2b — Mini-Transformer Encoder *(outlook)*

Self-attention from scratch, positional encoding.

### Stage 3 — Extended punctuation *(outlook)*

Question marks, exclamation marks, semicolons. Reuses all infrastructure.

### Stage 4 — Multi-task *(outlook)*

Joint prediction of punctuation + capitalisation (truecasing).

### Stage 5 — Deploy as a REST service *(optional)*

Octave trains, Java serves. Export weights from `.mat` to JSON or binary; build a Spring Boot inference endpoint — natural extension given the author's day-to-day stack.

---

## Evaluation

All stages are evaluated on the same held-out test set. Reported metrics:

- **Precision, Recall, F1 per class** (NONE, COMMA, PERIOD)
- **Macro-F1** — primary metric (unweighted mean F1 across classes)
- **Confusion matrix** 3×3
- Accuracy — context only, not the primary signal

---

## Known Pitfalls

| Pitfall | Mitigation |
|---------|------------|
| NONE dominates (~85%) | Weighted cross-entropy — mandatory |
| Train/test phrase leak | Split by document, not by sentence |
| Off-by-one at document boundary | Use `<PAD>` tokens or drop boundary windows |
| Dead ReLU units | Xavier/He init, not `N(0,1)` |
| Gradient explosion | Gradient clipping (good hygiene even in MLP) |

---

## Data Sources

- [Wolne Lektury](https://wolnelektury.pl/katalog/) — free Polish literary texts, `.txt` downloads
- [Wolne Lektury API](https://wolnelektury.pl/api/) — programmatic access

---

## For AI Agents

- **Runtime:** GNU Octave (`octave-cli`), MATLAB-syntax compatible.
- **`.mat` files are committed** — pre-computed data is stored in the repo for reproducibility.
- **Data flow:** `data/raw/*.txt` → `src/preprocess.m` → `data/processed/data.mat` (exports `words`, `labels`).
- **Tokenization:** lowercase full text → strip all chars except Polish letters (`a-ząćęłńóśźż`), whitespace, `,`, `.` → split on whitespace. Punctuation stays attached to preceding word (e.g. `"dom,"`, `"koniec."`).
- **Gotcha:** `preprocess.m` has a hardcoded `books` cell array — edit it to change source files.
- **Notes:** `notes/learning-plan.md` (5-stage curriculum), `notes/stage-0-bigram-baseline.md` (theory + implementation reference for Stage 0).

---

## Philosophy

*"Intentionally written in pure Octave, using only elementary arithmetic operations."*

No `torch`, no `sklearn`, no `autograd`. Every weight update, every gradient, every loss — written as a matrix expression. The goal is understanding, not benchmarks.
