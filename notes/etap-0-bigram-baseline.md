# Stage 0, Step 3 — Bigram Baseline Predictor

## Goal

Build the simplest possible punctuation predictor using only corpus statistics — no weights, no neural network, no learning. This is the **floor**: every subsequent model must beat it.

---

## Terminology

**Bigram** — a pair of two consecutive words. "stał na" is a bigram. The model looks at 2 words to predict the punctuation after the first one.

**N-gram** — the general term. A bigram is a 2-gram. A trigram is a 3-gram. We use bigrams here.

**Sliding window** — we move a fixed-size window one step at a time across the corpus:

```
"Stary zamek stał na wzgórzu i patrzył"

Window 1: ("Stary",   "zamek")   → label: NONE
Window 2: ("zamek",   "stał")    → label: NONE
Window 3: ("stał",    "na")      → label: NONE
Window 4: ("na",      "wzgórzu") → label: COMMA
Window 5: ("wzgórzu", "i")       → label: NONE
Window 6: ("i",       "patrzył") → label: PERIOD
```

Each position produces one training example. The label always belongs to `w_i` (the left word). `w_{i+1}` is context.

---

## Input / Output

**Input (training):** list of triples `(w_i, w_{i+1}, label)` from the training corpus.

**Input (prediction):** a pair of consecutive words `(w_i, w_{i+1})`.

**Output (prediction):** one label — `NONE=0`, `COMMA=1`, `PERIOD=2`.

---

## Model: Counting + Argmax

For each bigram, count how many times each label followed it in the corpus:

```
count("stał", "na", NONE)   = 45
count("stał", "na", COMMA)  = 3
count("stał", "na", PERIOD) = 1
```

Prediction = label with the highest probability:

```
P(label | w_i, w_{i+1}) = count(w_i, w_{i+1}, label) / count(w_i, w_{i+1}, *)
```

Where `count(w_i, w_{i+1}, *)` = total occurrences of this bigram across all labels.

---

## Problem: Unseen Bigrams

Test data will contain bigrams never seen during training. Count = 0, denominator = 0 — undefined.

### Laplace Smoothing (additive smoothing)

Add 1 to every label counter before computing probability. Add K to the denominator (K = number of classes = 3):

```
P(label | w_i, w_{i+1}) = (count(w_i, w_{i+1}, label) + 1)
                         / (count(w_i, w_{i+1}, *)     + K)
```

For an unseen bigram:
```
P(NONE   | "nowe", "słowo") = (0 + 1) / (0 + 3) = 0.33
P(COMMA  | "nowe", "słowo") = (0 + 1) / (0 + 3) = 0.33
P(PERIOD | "nowe", "słowo") = (0 + 1) / (0 + 3) = 0.33
```

Equal probability for all classes — honest admission of ignorance.

For a seen bigram, smoothing slightly pulls probabilities toward uniform:
```
Before: P(NONE | "stał", "na") = 45/49 = 0.918
After:  P(NONE | "stał", "na") = 46/52 = 0.885   ← small change
```

The effect is small for frequent bigrams, large for rare ones.

---

## Data Split

Split by **documents**, not by sentences — to prevent phrase leakage between sets.

```
Lalka, Quo Vadis, ...    → train  (80%)
Przedwiośnie             → val    (10%)  ← untouched until Stage 1
Potop                    → test   (10%)
```

If you split by sentences, the phrase "stał na wzgórzu" might appear in both train and test — the model has already seen it, the result is artificially good. This is called **data leakage**.

The validation set is not used here. Save it for Stage 1.

---

## Evaluation

Evaluate on the **test set only**.

### Why Not Accuracy

Class distribution in Polish literary text:
- NONE ≈ 85%
- COMMA ≈ 10%
- PERIOD ≈ 5%

A model that always predicts NONE achieves ~85% accuracy. It is useless. Accuracy does not detect this.

### Confusion Matrix 3×3

```
                  Predicted
                  NONE   COMMA  PERIOD
Actual  NONE   [  ?       ?       ?  ]
        COMMA  [  ?       ?       ?  ]
        PERIOD [  ?       ?       ?  ]
```

Diagonal = correct predictions. Off-diagonal = where the model makes systematic errors.

### TP / FP / FN / TN (per class, treated as binary)

For each class (e.g. COMMA vs. rest):

| | Predicted COMMA | Predicted not-COMMA |
|---|---|---|
| **Actually COMMA** | TP | FN |
| **Actually not-COMMA** | FP | TN |

- **TP** (True Positive) — model said COMMA, was right
- **FP** (False Positive) — model said COMMA, was wrong
- **FN** (False Negative) — model said not-COMMA, missed a real COMMA
- **TN** (True Negative) — model said not-COMMA, was right

### Precision, Recall, F1

```
Precision = TP / (TP + FP)     "how trustworthy are positive predictions?"
Recall    = TP / (TP + FN)     "how many real positives did we find?"
F1        = 2 * P * R / (P + R)
```

F1 uses the **harmonic mean** of Precision and Recall. The harmonic mean punishes imbalance — if one of them is near zero, F1 is near zero too. A model cannot hide behind high Precision with near-zero Recall (or vice versa).

```
P = 1.0, R = 0.01 → arithmetic mean = 0.505 (misleading)
                  → harmonic mean   = 0.020 (honest)
```

### Macro-F1

Simple average of F1 across all classes. Each class counts equally, regardless of how many examples it has:

```
Macro-F1 = (F1_NONE + F1_COMMA + F1_PERIOD) / 3
```

A model always predicting NONE:
- F1_NONE ≈ 1.0
- F1_COMMA = 0.0
- F1_PERIOD = 0.0
- Macro-F1 ≈ 0.33

---

## What to Report

Save results in `notes/etap-0-wyniki.md`:

- Confusion matrix
- Precision / Recall / F1 per class
- Macro-F1
- Accuracy (as context only)

---

## Expected Results

| Class  | Expected F1     |
|--------|-----------------|
| NONE   | high (~0.9)     |
| COMMA  | 0.15 – 0.30     |
| PERIOD | 0.40 – 0.55     |
| **Macro-F1** | **~0.50** |

These are weak results — that is the point. Stage 1 MLP must beat this by at least 10 percentage points of Macro-F1.
