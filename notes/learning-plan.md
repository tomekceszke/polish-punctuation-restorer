# Learning Plan: Polish Punctuation Restorer

Educational project. Goal: understand how to build a sequential classifier from scratch, deriving the math and implementing backprop manually on matrices.

---

## Reference Points from Your Experience

Throughout this plan I deliberately refer to things you have already done — so this project is a continuation of your path, not a fresh start.

| Source | What carries over |
|--------|-------------------|
| **`ml-applications`** (your atlas repo) | Most of the theoretical notes are already there: cost function, gradient descent, feature scaling, λ-regularisation, bias/variance, sigmoid, one-vs-all, backprop. In `notes/` — link rather than rewrite. Only add the deltas for softmax+CE and embeddings. |
| **`vehicles-counter`** | **Direct structural template.** Octave, single hidden layer network, backprop, weights saved as `Theta1.mat`/`Theta2.mat`, `learn.m`/`check.m`/`detect.m` split, `bin/conf/doc/lib/datasource` hierarchy. "Intentionally written in pure Matlab language, using only elementary arithmetic operations" — identical philosophy continued here. |
| **`traffic-light-detection`** | Same skeleton as vehicles-counter but with logistic regression + gradient descent. Reference for how you previously stepped from a simpler model (log-reg) to a more complex one (NN). Here you make the analogous jump: n-gram baseline → MLP. |
| **`car-price-prediction`** | Feature engineering on real-world data (otomoto scraping). Your experience with "dirty" data will be useful when preprocessing Wolne Lektury. |
| **`ml-login`** | "No external ML libraries, all algorithms written using only basic math formulas" — same philosophy kept here. Also shows you can build ML as a separate service with a clean interface (train/predict split) — an architectural pattern worth keeping even in Octave. |
| **Andrew Ng ML (Coursera)** | Especially **ex3** (forward pass, multi-class classification) and **ex4** (`nnCostFunction.m`, `sigmoidGradient.m`, `randInitializeWeights.m`, `checkNNGradients.m`). 80% of the backprop from ex4 is reusable — differences are listed at each step below. |

**Key conceptual difference** vs ex4/vehicles-counter/traffic-light-detection: there the **input was pixels** (continuous values in `[0,1]`), here the **input is word indices** (discrete values in `[1, V]`). This requires one additional layer — **embedding lookup** — which none of your previous projects had. This is the only new implementation concept to master. Everything else (cost, backprop, gradient check, optimizer) is a continuation of what you have done before.

**Second difference** vs ex4: there the output layer uses sigmoid + K output units (one-hot via one-vs-all). Here we use **softmax** + cross-entropy — the only mathematical change that requires its own derivation (the rest of the gradients are identical).

---

## Scope and Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Task | Token classification | For each word we predict the label of the punctuation that follows it |
| Labels | `{NONE=0, COMMA=1, PERIOD=2}` | Minimal meaningful set |
| Corpus | Wolne Lektury (PL) | Clean literary text, free to use, sufficient for a prototype |
| Stack | GNU Octave | `brew install octave`, MATLAB-like syntax |
| Representation | Embeddings over top-N words + `<UNK>` | Lookup = row indexing into a matrix |
| Context | Window ±k words | Start with k=2, tune later |
| Tokenisation | Whitespace + lowercase | Keep diacritics |
| Split | 80/10/10 by documents | Avoid phrase leakage |

---

## Repository Structure

Following your convention from `vehicles-counter` and `traffic-light-detection` (`learn.m`/`check.m`/`detect.m` + `bin/conf/doc/lib/datasource`) — extended for NLP:

```
polish-punctuation-restorer/
├── data/
│   ├── raw/                 # downloaded texts from Wolne Lektury
│   └── processed/           # tokens + labels (CSV/MAT)
├── src/
│   ├── preprocess.m         # raw text → (tokens, labels)
│   ├── vocab.m              # build top-N vocabulary
│   ├── baseline_ngram.m     # Stage 0
│   ├── mlp_forward.m        # Stage 1: forward pass
│   ├── mlp_backward.m       # Stage 1: gradients by hand
│   ├── learn.m              # training loop (equivalent of vehicles-counter/learn.m)
│   ├── check.m              # evaluation on test set (equivalent of check.m)
│   ├── detect.m             # inference on arbitrary text (equivalent of detect.m)
│   └── evaluate.m           # precision/recall/F1 per class
├── Theta1.mat, Theta2.mat   # saved weights — 1:1 convention from vehicles-counter
├── E.mat                    # embedding matrix (new vs vehicles-counter)
├── vocab.mat                # word↔index dictionary (new)
├── notes/                   # math derivations
└── README.md
```

---

## Stage 0 — Statistical Baseline (1–2 evenings)

**Goal:** have a *floor* that every subsequent stage must beat. Zero ML. Analogous to how in `ml-applications` you first describe linear regression as a starting point before logistic regression and NN.

1. Download 5–10 novels from Wolne Lektury (e.g. Prus, Sienkiewicz, Żeromski). Simpler source than scraping otomoto from `car-price-prediction` — there you wrestled with HTML; here it is clean `.txt`.
2. Write `preprocess.m`:
   - read file, lowercase, remove everything except `[a-ząćęłńóśźż\s,.]`
   - iterate over tokens, build list of `(word, label)` where label = punctuation immediately after the word (or NONE)
   - save as `.mat` (matrices are easier in Octave than structs)
3. `baseline_ngram.m`: for each pair `(w_i, w_{i+1})` count `count[label | w_i, w_{i+1}]`. Prediction = argmax with Laplace smoothing. Analogous to "guess the median price on otomoto" from `car-price-prediction` — a no-model baseline for comparison.
4. Evaluate on test set: **report F1 per class**, not accuracy.

**What you will learn:** working with a corpus, Polish preprocessing pitfalls, awareness of class imbalance (expect ~85% NONE), difference between accuracy and F1.

**Expected result:** F1 for PERIOD ~0.4–0.55, for COMMA ~0.15–0.3. Weak, but that is the reference point.

---

## Stage 1 — MLP with Manual Backprop (2–4 weeks)

**Goal:** build a neural network from matrices, derive every gradient by hand.

### Architecture

```
input:  [w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}]   (word indices, 5 integers)
  ↓ embedding lookup (E ∈ R^{V×d})
embed:  5 vectors of d dimensions each
  ↓ concat
x:      R^{5d}
  ↓ W1 ∈ R^{h×5d}, b1 ∈ R^h, ReLU
h:      R^h
  ↓ W2 ∈ R^{3×h}, b2 ∈ R^3, softmax
y_hat:  R^3  (distribution over {NONE, COMMA, PERIOD})
```

Starting hyperparameters: `V=5000, d=50, h=128, k=2, batch=64, lr=0.01`.

### Implementation Steps

1. **Vocabulary** (`vocab.m`): top-V words + `<UNK>` + `<PAD>` (for document boundaries). New vs vehicles-counter — there was no vocabulary there because the input was pixels.
2. **Forward** (`mlp_forward.m`): returns `y_hat` and activation cache for backprop. Analogous to `predict.m` from Andrew Ng ex3 — same pattern `a1 → z2 → a2 → z3 → a3`, just with embedding lookup before `a1` and softmax instead of sigmoid at the end.
3. **Loss**: weighted cross-entropy. Class weights = inverse of class frequency (otherwise the model learns to always predict NONE). Compare with `nnCostFunction.m` from ex4 — there it was sum-of-log-losses for sigmoid outputs; here one summation over classes with softmax.
4. **Backward** (`mlp_backward.m`): derive and implement the gradient for every parameter. Key point: the gradient of softmax+CE comes out as `(y_hat - y_true)` — derive this yourself, do not copy it. In ex4/`vehicles-counter` you have exactly the same formula as `δ3 = a3 - y` (there for sigmoid+CE; that it comes out identically for softmax+CE is exactly what is worth seeing yourself). Then `δ2 = (Θ2' * δ3) .* ReLU'(z2)` — same skeleton as ex4, just with ReLU instead of sigmoidGradient.
5. **Gradient check**: compare analytical gradient against numerical `(L(θ+ε) - L(θ-ε)) / 2ε`. Do not proceed without passing this. Relative difference < 1e-6 on a sample of parameters. Adapt `checkNNGradients.m` from ex4 — the structure is 1:1, you only change the cost function being called and add a check for the embedding gradient.
6. **Weight initialisation**: Xavier/He instead of `randInitializeWeights.m` from ex4 (uniform `ε_init`). With ReLU this matters — justification in `notes/`.
7. **Optimizer**: first SGD with momentum, then Adam. Observe the difference in loss curves empirically. In `vehicles-counter` you used `fmincg` (advanced optimizer from ex4) — here we deliberately step back to SGD to see the learning mechanics, then step forward to Adam.
8. **Training loop** (`learn.m`): mini-batches, log train/val loss every epoch, early stopping. Save weights to `Theta1.mat`, `Theta2.mat`, `E.mat` — convention from vehicles-counter.
9. **Evaluation** (`check.m`, `evaluate.m`): same method as in Stage 0. Compare Macro-F1 directly.

### Math to Derive Yourself (in `notes/`)

In `ml-applications` you already have: hypothesis, cost function, gradient descent, feature scaling, regularisation, bias/variance, sigmoid, decision boundary, one-vs-all, backprop at a high level. Do not duplicate — link. Only add the deltas specific to this project:

- Softmax gradient in isolation: ∂softmax(z)_i / ∂z_j (Jacobian matrix, different for i=j and i≠j)
- Cross-entropy + softmax gradient together — the elegant simplification to `p - y` (same form as `δ3 = a3 - y` for sigmoid+CE from ex4 — worth seeing that this is not a coincidence)
- ReLU gradient: simple, but worth writing out the edge cases (difference vs sigmoidGradient from ex4)
- **Embedding gradient**: why this is a `scatter-add`, not a full matrix multiply — the one gradient you will not find in ex4 or ml-applications. Your own derivation is mandatory.
- Chain rule through the full graph — write it out step by step, analogous to the diagram from ex4

---

## Metrics and Evaluation

One `evaluate.m` function, the same test set for all stages.

Always report:
- **Precision, Recall, F1 per class** (NONE, COMMA, PERIOD)
- **Macro-F1** (mean F1 across classes — do not weight by class size)
- **Confusion matrix** 3×3
- **Accuracy** as context only, not as the primary metric

---

## Common Pitfalls

- **NONE dominance**: without class weights the model reaches ~85% accuracy by always predicting NONE. Class weights in CE are mandatory.
- **Sentence-level leakage**: if you split by sentences instead of documents, phrases from one chapter will appear in both train and test. Split by files.
- **Off-by-one indexing**: the label after the last word in a document — handle it (PAD or discard).
- **Octave SparseMatrix**: keep one-hot input as indices, not as a sparse matrix — row indexing is faster and clearer.
- **Weight initialisation**: Xavier/He, not random from N(0,1). Otherwise ReLU dies.
- **Gradient explosion in long documents**: not a problem in MLP, but worth adding gradient clipping for hygiene.

---

## Further Stages (outlook, decision after Stage 1)

After completing Stage 1 you have your own MLP-based punctuation restorer — extension map:

- **Stage 2a** — bi-LSTM: introduces BPTT, gradient through time, gates. Large conceptual jump. Your first contact with sequential memory — none of your previous projects had this.
- **Stage 2b** — mini-Transformer encoder: self-attention from scratch, positional encoding. Mathematically cleaner, simpler to implement than LSTM.
- **Stage 3** — beyond NONE/COMMA/PERIOD: question marks, exclamation marks, semicolons. Reusable infrastructure.
- **Stage 4** — multi-task: punctuation + capitalisation (truecasing) simultaneously.
- **Stage 5 (optional)** — deploy as a REST service, analogous to the architecture from `ml-login` (`collector-service`/`learning-service`/`validator-service`). Since you work in Spring Boot daily, this is a natural extension — Octave trains, Java serves (weights exported from `.mat` to JSON or binary).

Decision after Stage 1, based on what pulls you in.

---

## Data Sources

- **Wolne Lektury**: https://wolnelektury.pl/katalog/ — download `.txt` files, free licence
- **API**: https://wolnelektury.pl/api/ — can be downloaded programmatically

---

## Definition of Done

**Stage 0:** working `baseline_ngram.m` + F1 report on the test set in `notes/stage-0-results.md`.

**Stage 1:** gradient check passes, network trains stably, Macro-F1 > baseline Macro-F1 by at least 10 percentage points, derivations in `notes/` complete.
