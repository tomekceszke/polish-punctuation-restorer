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
| **Stage 0 — Preprocessing** | ✅ Done | `preprocess.m` runs, train/test split committed |
| **Stage 0 — Bigram baseline** | ✅ Done | Macro-F1 = 0.506, results in `notes/stage-0-results.md` |
| **Stage 1 — MLP + backprop** | ⬜ Planned | Gradient check passes, macro-F1 beats baseline by ≥10 pp |
| **Stage 2a — Bi-LSTM** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 2b — Mini-Transformer** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 3 — Extended punctuation** | ⬜ Outlook | — |
| **Stage 4 — Multi-task** | ⬜ Outlook | — |
| **Stage 5 — REST deploy** | ⬜ Optional | — |

## Current State

**Stage 0 — Complete.**

- `src/preprocess.m` tokenizes raw `.txt` files, strips non-Polish characters, emits `(word, label)` pairs.
- Corpus: Polish literary texts from [Wolne Lektury](https://wolnelektury.pl). Train/test split documented in `notes/stage-0-preprocess.md`.
- Output: `data/processed/train.mat`, `test.mat`, `vocab.mat` — committed for reproducibility.
- Bigram baseline: Macro-F1 = 0.506. Full results in `notes/stage-0-results.md`.

### Label encoding

| Label | Value | Meaning |
|-------|-------|---------|
| No punctuation | `1` | Plain word |
| Comma | `2` | Word followed by `,` |
| Period | `3` | Word followed by `.` |

---

## Project Structure

```
ppr/
├── data/
│   ├── raw/                 # Source .txt files (Wolne Lektury)
│   └── processed/
│       ├── train.mat        # train_words, train_labels (~90%)
│       ├── test.mat         # test_words, test_labels (~10%)
│       └── vocab.mat        # top-1000 words from train
├── src/
│   ├── preprocess.m         # Entrypoint: raw text → train/test split
│   ├── baseline_ngram.m     # Stage 0: bigram frequency baseline
│   ├── config/
│   │   └── settings.m       # Shared constants: C_TRAIN_BOOKS, C_TEST_BOOKS, C_CUT_OFF_WORDS
│   ├── lib/
│   │   ├── tokenize.m       # Lowercase, strip, split on whitespace
│   │   ├── labelize.m       # Attach labels, strip trailing punctuation
│   │   ├── process.m        # Load and process a single book file
│   │   ├── build_vocab.m    # Build top-N vocabulary from words
│   │   ├── get_word_indices.m # Map words to vocab indices (UNK → N+1)
│   │   └── metrics.m        # Confusion matrix, precision/recall/F1 per class
│   └── utils/
│       └── epub2txt.py      # Convert .epub → .txt (stdlib only)
├── notes/
│   ├── learning-plan.md          # Full learning curriculum (EN)
│   ├── plan-nauki.md             # Full learning curriculum (PL)
│   ├── stage-0-preprocess.md     # Preprocessing design + train/test split
│   ├── stage-0-bigram-baseline.md # Theory + implementation reference
│   └── stage-0-results.md        # Baseline results: confusion matrix, F1 per class
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

Output is written to `../data/processed/train.mat`, `test.mat`, and `vocab.mat`.

**VS Code:** install the *Octave Debugger* extension. `.vscode/launch.json` is configured to run the current file with `octave-cli`.

To change source texts, edit `C_TRAIN_BOOKS` and `C_TEST_BOOKS` in `src/config/settings.m`. To convert an `.epub` file first, run `src/utils/epub2txt.py`.

---

## Learning Roadmap

This project follows a 5-stage curriculum, building complexity incrementally.

### Stage 0 — Statistical Baseline ✅ Done

N-gram frequency model: for each word pair `(w_i, w_{i+1})`, predict the most common following punctuation using counts.

| Class  | Train F1 | Test F1 |
|--------|----------|---------|
| NONE   | 0.9220   | 0.9278  |
| COMMA  | 0.5114   | 0.4877  |
| PERIOD | 0.3187   | 0.1020  |
| Macro  | 0.5840   | 0.5058  |

Full results: `notes/stage-0-results.md`.

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
- **Data flow:** `data/raw/*.txt` → `src/preprocess.m` → `data/processed/train.mat` + `test.mat` + `vocab.mat`.
- **Tokenization:** lowercase full text → strip all chars except Polish letters (`a-ząćęłńóśźż`), whitespace, `,`, `.` → split on whitespace. Punctuation stays attached to preceding word (e.g. `"dom,"`, `"koniec."`).
- **Gotcha:** source files are configured via `C_TRAIN_BOOKS` and `C_TEST_BOOKS` in `src/config/settings.m`.
- **Notes:** `notes/learning-plan.md` (5-stage curriculum), `notes/stage-0-bigram-baseline.md` (theory + implementation reference for Stage 0).

---

## Philosophy

*"Intentionally written in pure Octave, using only elementary arithmetic operations."*

No `torch`, no `sklearn`, no `autograd`. Every weight update, every gradient, every loss — written as a matrix expression. The goal is understanding, not benchmarks.
