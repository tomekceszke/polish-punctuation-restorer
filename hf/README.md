---
language: pl
license: mit
pipeline_tag: token-classification
tags:
  - punctuation-restoration
  - polish
  - from-scratch
  - matlab
  - octave
  - educational
  - mlp
  - no-framework
metrics:
  - f1
datasets:
  - tomekceszke/polish-punctuation-corpus
model-index:
  - name: polish-punctuation-restorer
    results:
      - task:
          type: token-classification
          name: Punctuation restoration
        dataset:
          type: tomekceszke/polish-punctuation-corpus
          name: Polish Punctuation Corpus (Wolne Lektury)
          split: test
        metrics:
          - type: f1
            name: Test Macro-F1
            value: 0.6077
---

# Polish Punctuation Restorer — Stage 1 (MLP from scratch)

A Polish punctuation restorer built **from scratch in MATLAB/Octave** — no PyTorch, no autograd,
every gradient derived by hand on paper and implemented as matrix arithmetic. It currently sits at
**test Macro-F1 = 0.608** and is climbing, one architecture at a time.

Given a Polish word and its context, the model predicts what follows it: nothing, a comma, or a
period.

```
input:   nie wiem czy przyjdzie ale poczekam jeszcze chwilę
output:  Nie wiem, czy przyjdzie, ale poczekam jeszcze chwilę.

input:   kiedy wrócił do domu było już ciemno a w oknach paliły się światła
output:  Kiedy wrócił do domu było już ciemno, a w oknach, paliły się światła,
```

Both are real output. The second one is the honest half: a missing comma before the subordinate
clause, a spurious one after `oknach`, and no closing period — Macro-F1 0.608 looks exactly like
this.

## What this is — and what it is not

**It is** a working, honest implementation of a neural sequence classifier where every part is
visible: the embedding lookup, the ReLU, the softmax, the cross-entropy, the backward pass, the
weight update. ~295K parameters, all of them traceable to a line of matrix code you can read in a
minute.

**It is not** a competitor to transformer-based punctuation models. A fine-tuned HerBERT will beat
it, and that is expected — the point here is the path from raw matrices to a model that works, not
the leaderboard position. The number is published because it moves: 0.511 → 0.608 → next stage.

If you want state-of-the-art Polish punctuation, use a transformer. If you want to see what a
punctuation model looks like with the framework removed, this repo is for you.

## Architecture

```
INPUT (7 word indices — a ±3 word window)
  [w_i-3, w_i-2, w_i-1, w_i, w_i+1, w_i+2, w_i+3]
         |  lookup in E (5001 × 50)
         v
EMBEDDING (350 numbers)
         |  W1 (128 × 350) + b1, ReLU
         v
HIDDEN (128 numbers)
         |  W2 (3 × 128) + b2, softmax
         v
OUTPUT  [p_NONE, p_COMMA, p_PERIOD]
```

| Component | Shape | Parameters |
|---|---|---|
| Embedding `E` | 5001 × 50 (vocab + `<UNK>` row) | 250,050 |
| `W1` | 128 × 350 | 44,800 |
| `b1` | 128 | 128 |
| `W2` | 3 × 128 | 384 |
| `b2` | 3 | 3 |
| **Total** | | **~295,400** |

Hyperparameters: `V=5000, d=50, h=128, k=3, batch=64, lr=0.005, epochs=30, patience=5, α=0.5`.
Plain mini-batch SGD, He initialisation, early stopping on validation Macro-F1. Gradients verified
numerically against analytic ones (relative error < 1e-5).

## Results

Held-out test set (two books never seen in training or validation):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| NONE | 0.9175 | 0.9258 | 0.9216 |
| COMMA | 0.5744 | 0.4758 | 0.5205 |
| PERIOD | 0.3551 | 0.4112 | 0.3811 |
| **Macro** | | | **0.6077** |

The dominant tuning lever was **class-weight tempering** — `w ∝ (1/count)^α` with `α=0.5` instead
of full inverse frequency, trading excess rare-class recall for precision. Context radius and
learning rate gave only small gains; capacity was not the bottleneck.

## Progress log

Each row is a closed stage. The table grows; the metric is meant to grow with it.

| Stage | Date | Model | Test Macro-F1 | Δ |
|---|---|---|---|---|
| 0 | 2026-05-07 | Bigram frequency baseline | 0.5106 | — |
| 1 | 2026-07-01 | MLP, hand-written backprop | **0.6077** | +9.7 pp |
| 2 | planned | Bi-LSTM or mini-transformer encoder | — | — |

## Training data

11 Polish literary works from [Wolne Lektury](https://wolnelektury.pl) — public domain or Free Art
License 1.3, translations included. The processed word/label pairs are published as
[tomekceszke/polish-punctuation-corpus](https://huggingface.co/datasets/tomekceszke/polish-punctuation-corpus). Document-level split: 69.2% train / 20.0% validation / 10.7%
test, so no phrase leaks across the boundary. Full attribution, with translators and per-work
licences, is in [`SOURCES.md`](SOURCES.md).

Tokenisation: lowercase, strip everything except Polish letters, whitespace, `,` and `.`, then
split on whitespace. Punctuation is taken off the word and becomes its label.

## Files

| File | What it is |
|---|---|
| `model.mat` | Weights in Octave's native format |
| `model_v7.mat` | Same weights as MATLAB v7 — readable by `scipy.io.loadmat` |
| `vocab.mat` / `vocab.txt` | 5000-word vocabulary; line number = model index, `<UNK>` = 5001 |
| `inference/` | Minimal Octave inference bundle (no training code) |
| `SOURCES.md` | Corpus attribution and licences |

## Usage

### Octave

```bash
cd inference
octave-cli detect.m      # interactive: type a sentence, get it punctuated
```

### Python — the whole forward pass, no framework

```python
import re
import numpy as np
from scipy.io import loadmat

m = loadmat("model_v7.mat")
E, W1, b1, W2, b2 = m["best_E"], m["best_W1"], m["best_b1"].T, m["best_W2"], m["best_b2"].T
vocab = open("vocab.txt", encoding="utf-8").read().split("\n")[:-1]
index = {w: i for i, w in enumerate(vocab)}
UNK, K = len(vocab), 3          # 0-based; <UNK> is the last row of E

def restore(text):
    words = re.sub(r"[^a-ząćęłńóśźż\s]", "", text.lower()).split()
    ids = [UNK] * K + [index.get(w, UNK) for w in words] + [UNK] * K
    windows = np.array([ids[i - K:i + K + 1] for i in range(K, len(ids) - K)])
    x = E[windows].reshape(len(words), -1)      # embedding lookup, (N, 350)
    h = np.maximum(0, x @ W1.T + b1)            # linear + ReLU, (N, 128)
    pred = (h @ W2.T + b2).argmax(1)            # logits -> class; softmax is monotone
    return " ".join(w + ["", ",", "."][p] for w, p in zip(words, pred))

print(restore("nie wiem czy przyjdzie ale poczekam jeszcze chwilę"))
# nie wiem, czy przyjdzie, ale poczekam jeszcze chwilę
```

That is the entire model at inference time: one lookup, one ReLU, one argmax. Capitalisation and
the closing period come from `post_process.m` in the Octave bundle, not from the model — the
snippet above is the raw prediction.

## Limitations

- Only two marks: comma and period. Question marks, exclamation marks and semicolons are a later stage.
- Trained on 19th–20th century literary prose. Contemporary, technical or spoken Polish is out of distribution.
- 5000-word vocabulary; everything else maps to `<UNK>`.
- Case is folded during tokenisation. The Octave bundle restores capitalisation with a rule
  (`post_process.m`), not with the model — truecasing is a planned stage.
- Periods are the weakest class (F1 0.3811): the model over-predicts sentence ends.

## Links

- **Training corpus:** <https://huggingface.co/datasets/tomekceszke/polish-punctuation-corpus>
- **Code and full write-up:** <https://github.com/tomekceszke/polish-punctuation-restorer>
- **Project page:** <https://tomek.ceszke.com/polish-punctuation-restorer/>
- **Derivations and stage notes:** [`notes/`](https://github.com/tomekceszke/polish-punctuation-restorer/tree/main/notes)
- **Paper (work in progress):** [`paper/paper.md`](https://github.com/tomekceszke/polish-punctuation-restorer/blob/main/paper/paper.md)

## Citation

```bibtex
@misc{ceszke2026ppr,
  author = {Tomasz Ceszke},
  title  = {Polish Punctuation Restorer: a from-scratch MLP in MATLAB/Octave},
  year   = {2026},
  url    = {https://github.com/tomekceszke/polish-punctuation-restorer}
}
```

MIT for code and weights. Corpus licences are listed in `SOURCES.md`.
