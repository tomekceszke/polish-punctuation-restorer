# Polish Punctuation Restorer

🌐 **Website:** [tomek.ceszke.com/polish-punctuation-restorer](https://tomek.ceszke.com/polish-punctuation-restorer/)

An educational project — building a sequential punctuation classifier from scratch in GNU Octave, deriving all math by hand and implementing backprop manually on matrices. No external ML libraries.

**Task:** given a Polish word, predict the punctuation mark that follows it: none, comma, or period.

**Model type:** sequence labeling / token classification (NLP task: punctuation restoration).

**Stage 1 model:** shallow feedforward neural network — an MLP with a single ReLU hidden layer over learned embeddings, ~295K parameters. **Result: test Macro-F1 = 0.608** vs bigram baseline 0.511.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Embedding matrix E | 5001 × 50 (V + `<UNK>` row) | 250,050 |
| W1 | 128 × 350 | 44,800 |
| b1 | 128 | 128 |
| W2 | 3 × 128 | 384 |
| b2 | 3 | 3 |
| **Total** | | **~295,400** |

For reference: GPT-2 small has 117M parameters — roughly 400× more.

---

## Contents

- [Paper](#paper)
- [Progress](#progress)
- [Current State](#current-state)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Learning Roadmap](#learning-roadmap)
- [Evaluation](#evaluation)
- [Known Pitfalls](#known-pitfalls)
- [Data Sources](#data-sources)
- [For AI Agents](#for-ai-agents)
- [AI Assistance & Methodology](#ai-assistance--methodology)
- [Philosophy](#philosophy)

---

## Paper

A standards-conforming academic write-up of this project — goal, methodology, results, and conclusions — lives in [`paper/paper.md`](paper/paper.md) (work in progress; build instructions in [`paper/README.md`](paper/README.md)).

---

## Progress

| Stage | Status | Done when |
|-------|--------|-----------|
| **Stage 0 — Preprocessing** | ✅ Done | `preprocess.m` runs, train/test split committed |
| **Stage 0 — Bigram baseline** | ✅ Done | Macro-F1 = 0.511, results in `notes/stage-0-results.md` |
| **Stage 1 — MLP + backprop** | ✅ Done | Macro-F1 = 0.608 (+9.7 pp over baseline, on par with the ≥10 pp target), results in `notes/stage-1-mlp.md` |
| **Stage 2a — Bi-LSTM** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 2b — Mini-Transformer** | ⬜ Outlook | Decision after Stage 1 |
| **Stage 3 — Extended punctuation** | ⬜ Outlook | — |
| **Stage 4 — Multi-task** | ⬜ Outlook | — |
| **Stage 5 — REST deploy** | ⬜ Optional | — |

## Current State

**Stage 0 — Complete. Stage 1 — Complete.**

- `src/preprocess.m` tokenizes raw `.txt` files, strips non-Polish characters, emits `(word, label)` pairs.
- Corpus: Polish literary texts from [Wolne Lektury](https://wolnelektury.pl). Document-level train/val/test split (69.2% / 20.0% / 10.7%) documented in `notes/stage-0-preprocess.md`.
- Output: `data/processed/train.mat`, `val.mat`, `test.mat`, `vocab.mat` — committed for reproducibility.
- Bigram baseline: Macro-F1 = 0.511 (V=5000, trained on train+val — it needs no validation set). Full results in `notes/stage-0-results.md`.
- MLP (`src/train.m`): mini-batch SGD with early stopping on validation Macro-F1; best weights committed as `data/processed/model.mat`. **Test Macro-F1 = 0.608** — the dominant tuning lever was class-weight tempering (`α=0.5`). Full results in `notes/stage-1-mlp.md`.

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
│       ├── train.mat        # train_words, train_labels (7 books, ~69%)
│       ├── val.mat          # val_words, val_labels (2 books, ~20%)
│       ├── test.mat         # test_words, test_labels (2 books, ~11%)
│       ├── vocab.mat        # top-5000 words from train
│       └── model.mat        # best Stage 1 MLP weights (E, W1, b1, W2, b2)
├── src/
│   ├── preprocess.m         # Entrypoint: raw text → train/test split
│   ├── baseline_ngram.m     # Stage 0: bigram frequency baseline
│   ├── mlp_init.m           # Stage 1: parameter initialization (E, W1, b1, W2, b2)
│   ├── mlp_forward.m        # Stage 1: forward pass with activation cache
│   ├── mlp_loss.m           # Stage 1: weighted cross-entropy loss
│   ├── mlp_backward.m       # Stage 1: manual backprop (all gradients)
│   ├── train.m              # Stage 1: SGD training loop, early stopping on val Macro-F1
│   ├── check.m              # Stage 1: test-set evaluation (loads model.mat, reports Macro-F1)
│   ├── config/
│   │   └── settings.m       # Shared constants: book lists + hyperparameters (C_V, C_K, C_LR, C_ALPHA, …)
│   ├── lib/
│   │   ├── tokenize.m       # Lowercase, strip, split on whitespace
│   │   ├── labelize.m       # Attach labels, strip trailing punctuation
│   │   ├── process.m        # Load and process a single book file
│   │   ├── build_vocab.m    # Build top-N vocabulary from words
│   │   ├── get_word_indices.m # Map words to vocab indices (UNK → N+1)
│   │   ├── build_windows.m  # Build ±k context windows of word indices
│   │   └── metrics.m        # Confusion matrix, precision/recall/F1 per class
│   ├── tests/               # Smoke tests + numerical gradient check (run in CI)
│   └── utils/
│       └── epub2txt.py      # Convert .epub → .txt (stdlib only)
├── notes/
│   ├── learning-plan.md          # Full learning curriculum
│   ├── stage-0-preprocess.md     # Preprocessing design + train/test split
│   ├── stage-0-bigram-baseline.md # Theory + implementation reference
│   ├── stage-0-results.md        # Baseline results: confusion matrix, F1 per class
│   └── stage-1-mlp.md            # MLP architecture, formulas, gradient check
├── paper/                        # Academic write-up (WIP)
```

Planned additions (per learning plan):

```
src/
└── detect.m                 # Inference on arbitrary text
```

---

## Getting Started

**Requirements:** GNU Octave (`brew install octave` on macOS).

```bash
# Run preprocessing (from src/)
cd src
octave-cli preprocess.m

# Train the Stage 1 MLP (writes ../data/processed/model.mat)
octave-cli train.m

# Evaluate the trained model on the test set
octave-cli check.m
```

Preprocessing output is written to `../data/processed/train.mat`, `val.mat`, and `test.mat`; `train.m` builds `vocab.mat` from the train set if it is missing.

**VS Code:** install the *Octave Debugger* extension. `.vscode/launch.json` is configured to run the current file with `octave-cli`.

To change source texts, edit `C_TRAIN_BOOKS`, `C_VAL_BOOKS`, and `C_TEST_BOOKS` in `src/config/settings.m`. To convert an `.epub` file first, run `src/utils/epub2txt.py`.

---

## Learning Roadmap

This project follows a 5-stage curriculum, building complexity incrementally.

### Stage 0 — Statistical Baseline ✅ Done

N-gram frequency model: for each word pair `(w_i, w_{i+1})`, predict the most common following punctuation using counts.

| Class  | Train F1 | Test F1 |
|--------|----------|---------|
| NONE   | 0.9403   | 0.9255  |
| COMMA  | 0.6492   | 0.4599  |
| PERIOD | 0.5580   | 0.1463  |
| Macro  | 0.7159   | 0.5106  |

V=5000, bigram argmax. Full results: `notes/stage-0-results.md`.

### Stage 1 — MLP with Hand-Written Backprop ✅ Done

Architecture: embedding lookup → linear+ReLU → linear → softmax. All gradients derived by hand, verified numerically.

```
INPUT (7 word indices)
  [w_{i-3}, w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}, w_{i+3}]
         |
         | lookup in E (5001 × 50)
         ↓
EMBEDDING (350 numbers)
         |
         | W1 (128 × 350) + b1 + ReLU
         ↓
HIDDEN LAYER (128 numbers)
         |
         | W2 (3 × 128) + b2 + softmax
         ↓
OUTPUT  [p_NONE, p_COMMA, p_PERIOD]
```

~295,400 parameters. Final hyperparameters: `V=5000, d=50, h=128, k=3, batch=64, lr=0.005, epochs=30, patience=5, α=0.5`. Full notes: `notes/stage-1-mlp.md`.

Results on the test set:

| Class  | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| NONE   | 0.922     | 0.922  | 0.922 |
| COMMA  | 0.588     | 0.467  | 0.520 |
| PERIOD | 0.333     | 0.442  | 0.381 |
| **Macro** | | | **0.608** |

| Model | Test Macro-F1 |
|-------|---------------|
| Stage 0 bigram baseline | 0.5106 |
| Stage 1 MLP (untuned) | 0.5354 |
| Stage 1 MLP (tuned) | **0.6077** |

+9.7 pp over the baseline — on par with the ≥10 pp target (within ±0.01 run-to-run init noise). The dominant tuning lever was **class-weight tempering**: `w ∝ (1/count)^α` with `α=0.5` instead of full inverse frequency, which traded excess rare-class recall for precision. Context radius and learning-rate changes gave only small gains; model capacity was not the bottleneck.

Key implementation steps (all complete):
1. Vocabulary with `<UNK>` (index V+1); boundary windows are dropped rather than padded.
2. Forward pass with activation cache.
3. Weighted cross-entropy loss — tempered inverse-frequency class weights `(N/(c·count))^α`.
4. Backward pass — all gradients derived by hand, including embedding scatter-add.
5. Gradient check: numerical vs analytic relative error < 1e-5.
6. He weight initialisation.
7. Plain mini-batch SGD (momentum/Adam turned out unnecessary at this scale).
8. Training loop (`src/train.m`) with early stopping on validation Macro-F1; best weights saved to `data/processed/model.mat`.

Math derived (in `notes/`): softmax Jacobian, softmax + CE simplification to `p − y`, ReLU gradient, embedding gradient (scatter-add), full chain-rule graph.

Done when: gradient check passes ✓, macro-F1 beats baseline by ≥10 pp ✓ (+9.7 pp, within noise of the target — accepted, see `notes/stage-1-mlp.md`).

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
| NONE dominates (80.6% of train tokens) | Weighted cross-entropy — mandatory; temper the weights (`α=0.5`), full inverse frequency over-predicts punctuation |
| Train/test phrase leak | Split by document, not by sentence |
| Off-by-one at document boundary | Drop boundary windows (chosen) or use `<PAD>` tokens |
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
- **Data flow:** `data/raw/*.txt` → `src/preprocess.m` → `data/processed/train.mat` + `val.mat` + `test.mat`; `src/train.m` → `vocab.mat` (built from train if missing) + `model.mat`.
- **Gotcha:** `vocab.mat` is built from the 7-book train set (post-validation-carve-out). The bigram baseline instead trains on train+val and rebuilds its vocab in-memory — do not "fix" it to use the committed `vocab.mat`.
- **Tokenization:** lowercase full text → strip all chars except Polish letters (`a-ząćęłńóśźż`), whitespace, `,`, `.` → split on whitespace. Punctuation stays attached to preceding word (e.g. `"dom,"`, `"koniec."`).
- **Gotcha:** source files are configured via `C_TRAIN_BOOKS`, `C_VAL_BOOKS`, and `C_TEST_BOOKS` in `src/config/settings.m`; hyperparameters live there too.
- **Notes:** `notes/learning-plan.md` (5-stage curriculum), `notes/stage-0-bigram-baseline.md` (theory + implementation reference for Stage 0), `notes/stage-1-mlp.md` (MLP architecture, formulas, results).

---

## AI Assistance & Methodology

This project is built with AI assistance (Claude, via Claude Code), and the way that help works is deliberately constrained. The point of the project is to *understand* how a neural sequence classifier works — so the AI is never allowed to do the understanding for me.

These are hard rules, and they are never broken:

- **Socratic only** — the AI asks guiding questions and gives pointers, not answers. It nudges; it does not solve.
- **No ready-to-paste code** — the AI never writes the implementation. I write every line myself.
- **I derive the math** — every formula, every gradient, every design decision is worked out by hand. The AI explains concepts and checks my reasoning, but the derivation is mine.

The goal is understanding, not throughput. Handing over code would defeat the entire purpose of the exercise — see [Philosophy](#philosophy) below.

**Illustrative exchange** (one Socratic turn — note: no code is given):

> **Me:** How do I accumulate the embedding gradient when a word appears at more than one window position?
>
> **AI:** Good — you've spotted the case that trips most people up. Start with one question: if the same row of `E` is touched by two positions, and you *assign* its gradient instead of *adding*, what happens to the first contribution? Once you've answered that, what Octave operator gives you the behaviour you actually want?

---

## Philosophy

*"Intentionally written in pure Octave, using only elementary arithmetic operations."*

No `torch`, no `sklearn`, no `autograd`. Every weight update, every gradient, every loss — written as a matrix expression. The goal is understanding, not benchmarks.
