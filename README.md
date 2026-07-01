# Polish Punctuation Restorer

🌐 **Website:** [tomek.ceszke.com/polish-punctuation-restorer](https://tomek.ceszke.com/polish-punctuation-restorer/)

An educational project — building a sequential punctuation classifier from scratch in GNU Octave, deriving all math by hand and implementing backprop manually on matrices. No external ML libraries.

**Task:** given a Polish word, predict the punctuation mark that follows it: none, comma, or period.

**Model type:** sequence labeling / token classification (NLP task: punctuation restoration).

**Stage 1 model:** shallow feedforward neural network — an MLP with a single ReLU hidden layer over learned embeddings, ~282K parameters.

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
| **Stage 1 — MLP + backprop** | 🔄 In Progress | Gradient check passes, macro-F1 beats baseline by ≥10 pp |
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
- Bigram baseline: Macro-F1 = 0.511 (V=5000). Full results in `notes/stage-0-results.md`.

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
│       └── vocab.mat        # top-5000 words from train
├── src/
│   ├── preprocess.m         # Entrypoint: raw text → train/test split
│   ├── baseline_ngram.m     # Stage 0: bigram frequency baseline
│   ├── mlp_init.m           # Stage 1: parameter initialization (E, W1, b1, W2, b2)
│   ├── mlp_forward.m        # Stage 1: forward pass with activation cache
│   ├── mlp_loss.m           # Stage 1: weighted cross-entropy loss
│   ├── mlp_backward.m       # Stage 1: manual backprop (all gradients)
│   ├── train.m              # Stage 1: training loop (WIP)
│   ├── config/
│   │   └── settings.m       # Shared constants: C_TRAIN_BOOKS, C_TEST_BOOKS, C_V
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
| NONE   | 0.9403   | 0.9255  |
| COMMA  | 0.6492   | 0.4599  |
| PERIOD | 0.5580   | 0.1463  |
| Macro  | 0.7159   | 0.5106  |

V=5000, bigram argmax. Full results: `notes/stage-0-results.md`.

### Stage 1 — MLP with Hand-Written Backprop 🔄 In Progress

Architecture: embedding lookup → linear+ReLU → linear → softmax. All gradients derived by hand, verified numerically.

```
INPUT (5 word indices)
  [w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}]
         |
         | lookup in E (5000 × 50)
         ↓
EMBEDDING (250 numbers)
         |
         | W1 (128 × 250) + b1 + ReLU
         ↓
HIDDEN LAYER (128 numbers)
         |
         | W2 (3 × 128) + b2 + softmax
         ↓
OUTPUT  [p_NONE, p_COMMA, p_PERIOD]
```

~282,500 parameters. Starting hyperparameters: `V=5000, d=50, h=128, k=2, batch=64, lr=0.01`. Full notes: `notes/stage-1-mlp.md`.

Key implementation steps:
1. Build vocabulary with `<UNK>` and `<PAD>`.
2. Forward pass with activation cache.
3. Weighted cross-entropy loss (class weights = inverse frequency).
4. Backward pass — all gradients derived by hand, including embedding scatter-add.
5. Gradient check: numerical vs analytic relative error < 1e-5.
6. Xavier/He weight initialisation.
7. SGD with momentum → Adam; observe the difference empirically.
8. Training loop (`src/train.m`) with early stopping; save `Theta1.mat`, `Theta2.mat`, `E.mat`.

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
| NONE dominates (80.6% of train tokens) | Weighted cross-entropy — mandatory |
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
