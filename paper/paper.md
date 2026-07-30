---
title: "Polish Punctuation Restoration from Scratch: A Hand-Derived Neural Classifier in GNU Octave"
author: "Tomasz Ceszke"
affiliation: "Independent researcher"
date: "July 2026"
keywords: [punctuation restoration, sequence labeling, multilayer perceptron, word embeddings, backpropagation, Polish NLP, GNU Octave]
---

# Polish Punctuation Restoration from Scratch: A Hand-Derived Neural Classifier in GNU Octave

**Tomasz Ceszke** · Independent researcher · July 2026

Repository: <https://github.com/tomekceszke/polish-punctuation-restorer>

> **Status — Work in progress (Stage 1 of 5 complete).** This paper documents an
> ongoing educational project. Stage 0 (statistical baseline) and Stage 1 (a multilayer
> perceptron with hand-derived backpropagation) are complete and reported with final
> numbers: the tuned MLP reaches a **test macro-F1 of 0.608** against the baseline's
> 0.511. Sections describing Stages 2–5 are deliberately marked as outlook. The
> document will be revised as the project advances.

---

## Abstract

We present **Polish Punctuation Restorer (PPR)**, an educational system that predicts the
punctuation mark following each word of Polish text — *none*, *comma*, or *period* — framed
as a supervised token-classification (sequence-labeling) problem. The defining constraint of
the project is pedagogical: the entire model is implemented in **GNU Octave using only
elementary matrix arithmetic**, with every gradient derived by hand and verified numerically,
and **no external machine-learning libraries** (no autograd, no `torch`, no `sklearn`). We
describe a completed statistical baseline — a bigram frequency model that reaches a
**macro-F1 of 0.511** on a held-out test set — and a completed neural model: a
multilayer perceptron (MLP) with a single ReLU hidden layer over learned word embeddings,
roughly **295,400 parameters**, closely following the classic neural probabilistic language
model of Bengio et al. [1]. We report the corpus design (11 public-domain Polish literary texts split
by document to prevent phrase leakage), the manual derivation of the softmax + cross-entropy,
ReLU, and embedding-scatter-add gradients, and the gradient-checking protocol that validates
them. The tuned MLP reaches a **test macro-F1 of 0.608** — +9.7 percentage points over the
baseline, on par with the ≥10 pp target within run-to-run initialization noise. The single
most valuable tuning intervention was not capacity or optimization but **tempering the
inverse-frequency class weights** ($w \propto \text{count}^{-\alpha}$, $\alpha = 0.5$), which
rebalanced rare-class recall toward precision.

---

## 1. Introduction

### 1.1 Motivation

Most practical machine-learning work today is built on high-level frameworks that hide the
underlying mathematics behind automatic differentiation. That is excellent for shipping
products and poor for *understanding*. This project takes the opposite stance: its goal is not
a competitive benchmark score but a complete, first-principles understanding of how a small
neural sequence classifier works, obtained by deriving and implementing **every** operation —
the forward pass, the loss, each gradient, and the weight updates — as explicit matrix
expressions in GNU Octave. The guiding principle, stated in the project README, is:

> *"Intentionally written in pure Octave, using only elementary arithmetic operations."*
> No `torch`, no `sklearn`, no `autograd`. Every weight update, every gradient, every loss —
> written as a matrix expression. The goal is understanding, not benchmarks. (see
> [`../README.md`](../README.md))

### 1.2 Task definition

Given a sequence of Polish words, predict for each word the punctuation that immediately
follows it. We restrict the label set to three classes:

| Label | Value | Meaning |
|-------|-------|---------|
| NONE | 1 | word followed by no punctuation |
| COMMA | 2 | word followed by `,` |
| PERIOD | 3 | word followed by `.` |

This is **punctuation restoration**, a standard NLP task usually approached as sequence
labeling / token classification [10]. It is a natural didactic target: the input is text, the
output is a small discrete set, the classes are strongly imbalanced (so accuracy is misleading
and macro-F1 is required), and the problem scales smoothly from a trivial count-based baseline
to embeddings, recurrence, and attention.

### 1.3 Contributions (framed as learning goals)

1. A reproducible **statistical baseline** (bigram argmax) with a full evaluation, establishing
   the number any learned model must beat.
2. A from-scratch **MLP-over-embeddings** classifier whose forward and backward passes are
   derived by hand and implemented as plain matrix operations.
3. A **numerical gradient-checking** harness that validates each analytic gradient before any
   training is attempted — including the embedding *scatter-add* gradient that frameworks
   normally hide.
4. A **curriculum**: a five-stage roadmap that grows the same task from counting to sequential
   and attention-based models, reusing one data pipeline and one evaluation protocol throughout.

The companion learning curriculum is maintained in
[`../notes/learning-plan.md`](../notes/learning-plan.md).

---

## 2. Background and Related Work

**Punctuation restoration.** Restoring punctuation in unpunctuated text (e.g. ASR output) is
commonly cast as per-token classification and addressed with sequence models; Tilk and Alumäe
[10] is a representative neural treatment. We adopt the same framing but, at this stage, use a
fixed-context model rather than a recurrent one.

**N-gram baselines.** Before any learning, a frequency model that memorises the most likely
label for each context is a strong, honest baseline and a sanity check for the data pipeline.

**Word embeddings.** Rather than one-hot word inputs, we learn a dense vector per vocabulary
word, in the tradition of distributed representations [7]. The specific architecture — embed a
fixed window of words, concatenate, and feed a feedforward network — follows the neural
probabilistic language model of Bengio et al. [1], adapted here from next-word prediction to
symmetric-window classification, with ReLU in place of tanh.

**MLP, ReLU, softmax, cross-entropy.** The classifier is a feedforward network with a single
rectified-linear hidden layer [3, 4] and a softmax output [5] trained with cross-entropy [6].
Its parameters are learned by backpropagation [2].

**Initialization and optimization.** Weights of ReLU layers are initialized with the He scheme
[8] (Glorot/Xavier [9] for comparison); the optimizer is plain mini-batch SGD — momentum and
Adam [11] were held in reserve but proved unnecessary at this scale (§4.7).

---

## 3. Task and Data

### 3.1 Corpus

The corpus consists of **11 Polish-language literary texts** (public domain or free-licensed,
several in translation) from [Wolne Lektury](https://wolnelektury.pl). The full preprocessing design is documented in
[`../notes/stage-0-preprocess.md`](../notes/stage-0-preprocess.md).

| Set | Books | Tokens | Share |
|-----|-------|--------|-------|
| **Train** | Chłopi, Lalka, Nad Niemnem, Proces, Przedwiośnie, Moralność Pani Dulskiej, Mały Książę | 828,125 | 69.2% |
| **Validation** | Ziemia Obiecana, 1984 | 239,580 | 20.0% |
| **Test** | Syzyfowe Prace, Tajemniczy Ogród | 128,235 | 10.7% |

The validation set was carved out of the original 9-book training set (1,067,705 tokens)
once early stopping required one; the test set was never touched, so Stage 0 numbers remain
valid. The bigram baseline, which has no use for a validation set, trains on train+validation
— the original 1,067,705 tokens — keeping its reported figures canonical.

### 3.2 Tokenization and labeling

Preprocessing (`src/preprocess.m`) lowercases the text, removes every character except Polish
letters, whitespace, comma, and period (keeping the regex class `[a-ząćęłńóśźż ,.]`), and splits
on whitespace. The punctuation character immediately following a token becomes that token's
label and is then removed from the stream. Each token therefore yields a `(word, label)` pair.

### 3.3 Document-level split (anti-leakage)

The train/validation/test split is made **by document, not by sentence or token position**. An
entire book belongs to exactly one set. This prevents *phrase leakage* — the same multi-word phrase
appearing, with its punctuation, in both train and test — which would inflate the apparent
performance of any model that memorises contexts. The test set deliberately mixes original
Polish prose (*Syzyfowe Prace*, Żeromski — the same author appears in training via
*Przedwiośnie*) and a translation (*Tajemniczy Ogród*) for stylistic variety.

### 3.4 Class imbalance

The label distribution is severely skewed. Exact counts from the processed data give a
train+validation distribution of **80.58% / 12.14% / 7.29%** (NONE / COMMA / PERIOD; 1,067,705 tokens)
and a test distribution of **83.38% / 10.24% / 6.37%** (128,235 tokens). Because a degenerate
"always NONE" classifier would already score ~80.6% accuracy (and a macro-F1 of only ~0.33),
**accuracy is not a useful metric** and **macro-F1** (the unweighted mean of per-class F1) is
adopted as the primary metric throughout.

---

## 4. Methodology

### 4.1 Bigram baseline (Stage 0, complete)

The baseline is a pure frequency model with no learning. For every adjacent word pair
$(w_i, w_{i+1})$ in the training data it counts how often each label follows, then at prediction
time emits the most frequent label (argmax over counts) for that pair. Implementation:
`src/baseline_ngram.m`. Theory and design: [`../notes/stage-0-bigram-baseline.md`](../notes/stage-0-bigram-baseline.md).

### 4.2 MLP architecture (Stage 1, complete)

The neural model embeds a symmetric window of $2k+1$ words ($k = 3$, i.e. seven words centered on
the target), concatenates their embeddings, and classifies the center word's following
punctuation. The context radius started at $k = 2$ and was widened to $k = 3$ during tuning
(§5.2). Full notes: [`../notes/stage-1-mlp.md`](../notes/stage-1-mlp.md).

```
INPUT  [w_{i-3}, w_{i-2}, w_{i-1}, w_i, w_{i+1}, w_{i+2}, w_{i+3}]   (7 word indices)
   │  lookup in E (5001 × 50)
   ▼
EMBEDDING  (350 numbers = 7 × 50, concatenated)
   │  W1 (128 × 350) + b1, then ReLU
   ▼
HIDDEN LAYER  (128 numbers)
   │  W2 (3 × 128) + b2, then softmax
   ▼
OUTPUT  [p_NONE, p_COMMA, p_PERIOD]
```

| Parameter | Shape | Count | Initialization |
|-----------|-------|-------|----------------|
| $E$ (embeddings) | 5001 × 50 | 250,050 | `randn * 0.01` |
| $W_1$ | 128 × 350 | 44,800 | He [8]: `randn * sqrt(2/350)` |
| $b_1$ | 128 × 1 | 128 | zeros |
| $W_2$ | 3 × 128 | 384 | He [8]: `randn * sqrt(2/128)` |
| $b_2$ | 3 × 1 | 3 | zeros |
| **Total** | | **~295,400** | |

The embedding matrix has $V + 1$ rows: the extra row is the `<UNK>` vector, to which all
out-of-vocabulary words are mapped. During training and evaluation, windows that would cross a
document boundary are dropped rather than padded. At inference time (`src/detect.m`) this is
inverted: the index vector is padded with $k$ `<UNK>` entries at both ends, so that $n$ input words
yield exactly $n$ windows and no word — in particular not the last one, which carries the
sentence-final period — is left without a prediction.

For scale, GPT-2 small has ~117M parameters — about 400× more. This mirrors the architecture of
Bengio et al.'s neural probabilistic language model [1] — embed a fixed context window, concatenate,
and feed a feedforward network — adapted to classification. Hyperparameters: $V = 5000$, $d = 50$,
$h = 128$, $k = 3$.

### 4.3 Forward pass

For a batch of $N$ samples, $X_\text{idx} \in \mathbb{N}^{N \times 7}$:

$$
x = \text{concat}\big(E_{w_{i-3}}, \dots, E_{w_{i+3}}\big) \in \mathbb{R}^{N \times 350}, \quad
s_1 = x W_1^\top + b_1^\top, \quad
a_1 = \max(0, s_1),
$$
$$
s_2 = a_1 W_2^\top + b_2^\top, \quad
p_{ij} = \frac{e^{s_{2,ij}}}{\sum_{l} e^{s_{2,il}}}.
$$

The softmax [5] is computed in a numerically stable form by subtracting the per-row maximum from
$s_2$ before exponentiating. The forward pass returns `probs` together with a cache
($x$, $s_1$, $a_1$) for use in backpropagation. Implementation: `src/mlp_forward.m`.

### 4.4 Loss: weighted cross-entropy

To counter the class imbalance of §3.4, we use cross-entropy [6] weighted by *tempered*
inverse class frequency. With one-hot targets $y$ and per-class weights

$$
w_k = \left( \frac{N_\text{total}}{c \cdot N_k} \right)^{\alpha}, \qquad \alpha = 0.5,
$$

(so rare classes are up-weighted, but less aggressively than full inverse frequency):

$$
L = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \, \log p_{i, y_i}.
$$

Without weighting ($\alpha = 0$), the network collapses to always predicting NONE. Full
inverse frequency ($\alpha = 1$) overshoots the other way — it up-weights the rare classes
roughly 11:1 and the model over-predicts punctuation (high recall, low precision on COMMA and
PERIOD). Tempering with $\alpha = 0.5$ compresses the ratio to roughly 3:1 and was the single
largest contributor to the final score (§5.2). Implementation: `src/mlp_loss.m` (loss),
`src/train.m` (weight construction).

### 4.5 Backpropagation (derived by hand)

All gradients are derived analytically and written as matrix expressions. The key results:

- **Output layer.** The softmax + cross-entropy gradient simplifies to the classic difference of
  probabilities and one-hot target, scaled by the sample weight and batch size:
  $$ \delta_2 = (p - y) \odot \tfrac{w}{N} \in \mathbb{R}^{N \times 3}. $$
- **Second linear layer.** $\;dW_2 = \delta_2^\top a_1,\qquad db_2 = \sum_i \delta_{2,i}.$
- **ReLU hidden layer.** Backpropagate through the rectifier with its subgradient mask:
  $$ \delta_1 = \big(\delta_2 W_2\big) \odot \mathbb{1}[s_1 > 0]. $$
- **First linear layer.** $\;dW_1 = \delta_1^\top x,\qquad db_1 = \sum_i \delta_{1,i}.$
- **Embedding scatter-add.** The gradient w.r.t. the input embeddings, $\delta_1 W_1$, is split
  into $2k+1 = 7$ blocks of $d = 50$ columns (one per window position) and **accumulated** into the
  rows of $dE$ indexed by the corresponding words: `dE(idx, :) += δ_embed`. Using `+=`
  (accumulation) rather than `=` is essential — a word may appear at multiple positions.

Implementation: `src/mlp_backward.m`. The full chain-rule derivation is in
[`../notes/stage-1-mlp.md`](../notes/stage-1-mlp.md).

### 4.6 Gradient checking

Before any training, every analytic gradient is validated numerically. For each parameter
$\theta$ we compute the central finite difference $(L(\theta + \varepsilon) - L(\theta -
\varepsilon)) / 2\varepsilon$ with $\varepsilon = 10^{-5}$ and compare it to the analytic
gradient via relative error $|g_\text{analytic} - g_\text{num}| / (|g_\text{analytic}| +
|g_\text{num}|)$, requiring **< $10^{-5}$**. The check runs on a tiny model ($V=20$, $d=3$,
$h=4$, batch $=5$) for speed and covers $W_2$, $W_1$, $b_2$, $b_1$, and random entries of $E$
(validating the scatter-add). **Training does not begin until this passes.** Implementation and
test: `src/tests/test_grad_check.m`.

### 4.7 Training setup

Training (`src/train.m`) uses plain mini-batch SGD: batch size 64, learning rate 0.005, at most
30 epochs. Momentum and Adam [11] were planned as a comparison but turned out unnecessary at
this scale — plain SGD converges in minutes, and optimization was not the bottleneck (§5.2).

Early stopping is driven by **validation macro-F1**, not validation loss: after each epoch the
model is evaluated on the validation set (§3.1), and training stops after 5 epochs without
improvement (patience = 5). The two criteria genuinely disagree — the best-loss epoch and the
best-F1 epoch differ — and since macro-F1 is the judged metric, it decides. Validation loss is
still logged as a diagnostic. The best parameters seen are saved as a single file,
`data/processed/model.mat`, committed to the repository like the other `.mat` artifacts.

---

## 5. Results

### 5.1 Stage 0 — bigram baseline (final)

Setup: $V = 5000$, bigram argmax over training counts, no smoothing. Confusion matrix on the
test set:

```
              Predicted
              NONE     COMMA   PERIOD
Actual NONE  [105082    1114     731]
       COMMA [  8356    4473     304]
       PERIOD[  6716     731     727]
```

Per-class precision/recall/F1 (test) and the train/test F1 comparison:

| Class | Precision (test) | Recall (test) | F1 (test) | F1 (train) |
|-------|------------------|---------------|-----------|------------|
| NONE | 0.875 | 0.983 | 0.9255 | 0.9403 |
| COMMA | 0.708 | 0.341 | 0.4599 | 0.6492 |
| PERIOD | 0.413 | 0.089 | 0.1463 | 0.5580 |
| **Macro** | — | — | **0.5106** | **0.7159** |

Full results, including a $V=1000$ archive run, are in
[`../notes/stage-0-results.md`](../notes/stage-0-results.md).

For context, overall test accuracy is 0.860 — dominated by the NONE class, which is exactly why
macro-F1, not accuracy, is the primary metric (§3.4).

**Observations.** NONE is easy and stable (train ≈ test F1). COMMA generalizes moderately. PERIOD
shows a large train–test gap (0.558 → 0.146): the count model memorises period-ending bigrams
from training and fails on unseen ones — precisely the weakness a model with *generalising*
distributed representations should address. The baseline therefore sets the bar:

$$ \textbf{Macro-F1}_\text{baseline} = 0.511 \;\;\Rightarrow\;\; \textbf{Stage 1 target} > 0.611\ (+10\ \text{pp}). $$

### 5.2 Stage 1 — MLP (final)

Two snapshots are reported: the model straight after implementation (no tuning), and after a
small tuning round that shows what each intervention was worth.

**Untuned** ($k=2$, $\alpha=1$, lr 0.01, 10 epochs): test macro-F1 = **0.5354** — already +2.5 pp
over the baseline out of the box. The class weights work (no collapse to always-NONE; rare-class
recall is decent), but precision on COMMA and PERIOD is low: full inverse-frequency weighting
makes the model over-predict punctuation.

**Tuned** (final config of §4.2/§4.7: $k=3$, $\alpha=0.5$, lr 0.005, ≤30 epochs, patience 5),
test set:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| NONE | 0.922 | 0.922 | 0.922 |
| COMMA | 0.588 | 0.467 | 0.520 |
| PERIOD | 0.333 | 0.442 | 0.381 |
| **Macro** | — | — | **0.6077** |

| Model | Test Macro-F1 |
|-------|---------------|
| Stage 0 bigram baseline | 0.5106 |
| Stage 1 MLP (untuned) | 0.5354 |
| Stage 1 MLP (tuned) | **0.6077** |
| Target (baseline + 10 pp) | 0.611 |

The result is **+9.7 pp over the baseline — on par with the +10 pp target** (within run-to-run
noise of ±0.01 from random initialization). Contribution of each tuning step, applied in order:

| Step | Change | Macro-F1 |
|------|--------|----------|
| untuned | $\alpha=1$, $k=2$, lr 0.01, 10 epochs | 0.5354 (test) |
| optimizer | lr 0.01→0.005, epochs 10→30, patience 3→5 | ~0.555 (val) |
| context | $k$: 2→3 | ~0.559 (val) |
| **class weights** | $\alpha$: 1→0.5 | **0.6063 (test)** |
| final | retrain at $\alpha=0.5$ | **0.6077 (test)** |

The dominant lever by far was **tempering the class weights** (§4.4): a sweep over
$\alpha \in \{0.4, 0.5, 0.6\}$ peaked cleanly at 0.5, and the change lifted COMMA precision
from 0.33 to 0.59 (NONE rose too, from fewer false-positive commas). Learning-rate and context
changes gave only small gains; a diagnostic check of train vs. validation loss showed the model
was not capacity-starved, so hidden size and embedding dimension were left alone — rare-class
*precision*, not capacity, was the bottleneck.

Compared with the baseline per class (§5.1): PERIOD F1 improves from 0.146 to 0.381 — the
distributed representations generalize to unseen period-ending contexts exactly where the count
model memorizes and fails — and COMMA improves from 0.460 to 0.520, while NONE stays
essentially equal (0.926 vs 0.922).

---

## 6. Discussion

**Why weighted loss is mandatory — and why it must be tempered.** With ~80% of tokens labeled
NONE, an unweighted objective makes "always NONE" a strong local optimum (macro-F1 ≈ 0.33).
Inverse-frequency class weights force the network to attend to the rare COMMA and PERIOD
classes. But the weighting strength is itself a hyperparameter, and it proved to be the most
important one in the project (§5.2): full inverse frequency trades away too much precision for
recall, and tempering the weights with $\alpha = 0.5$ (§4.4) was worth more than every other
tuning change combined. This is the single most important design choice for the metric we
care about.

**Why split by document.** Splitting by document rather than by sentence eliminates phrase leakage
and yields an honest estimate of generalization. The baseline's PERIOD train–test gap is direct
evidence that leakage, if present, would have masked a real weakness.

**Known pitfalls and mitigations.** Dead ReLU units (mitigated by He initialization [8] rather
than $\mathcal{N}(0,1)$); NaN loss from too-large a learning rate or an unclamped log; gradient
explosion (clipping is good hygiene even for an MLP); and the most common backprop bug — a
scatter-add written with `=` instead of `+=`, which the gradient check is designed to catch.

**Limitations.** A fixed context window of seven words cannot capture long-range structure; the
label set is reduced to three classes; abbreviations such as "ul." or "dr." are treated as PERIOD,
a known and accepted simplification at this stage. These limitations directly motivate the
roadmap below.

---

## 7. Future Work

The project follows a five-stage curriculum
([`../notes/learning-plan.md`](../notes/learning-plan.md)); Stages 2–5 are outlook:

- **Stage 2a — Bi-LSTM.** Backpropagation through time, gates, true sequential memory.
- **Stage 2b — Mini-Transformer encoder.** Self-attention and positional encoding from scratch.
- **Stage 3 — Extended punctuation.** Add `?`, `!`, `;`, reusing the entire pipeline.
- **Stage 4 — Multi-task.** Joint punctuation + capitalization (truecasing).
- **Stage 5 — Deployment (optional).** Export weights from `.mat` to JSON/binary and serve
  inference from a Spring Boot endpoint (Octave trains, Java serves).

With Stage 1 complete, the immediate next decision is which sequential architecture to build
first: the Bi-LSTM (2a) or the mini-Transformer encoder (2b).

---

## 8. Use of AI Assistance & Methodology

This project was developed with the assistance of an AI system (Claude, via Claude
Code). In the interest of full disclosure, we state precisely what that assistance
did and — more importantly — did not include. The assistance was deliberately
constrained to a **Socratic** role: conceptual explanation, guiding questions, and
review of hand-derived mathematics. It **excluded the generation of implementation
code**. Every line of Octave was written by the author, and every result in §4.5 —
the softmax + cross-entropy, ReLU, and embedding scatter-add gradients — was
derived by hand, with the AI offering questions and sanity checks rather than
solutions. The from-scratch implementation and the hand-derivation of all gradients
(a core contribution, §1.3) are therefore the author's own work.

This constraint is a deliberate methodological choice in service of the project's
pedagogical goal (§1.1): the value of the work lies in the derivation, not in the
speed of producing code, and delegating either would defeat its purpose. A concrete
example of the interaction style is given in the repository README (see
[`../README.md`](../README.md)).

---

## 9. Conclusion

PPR is an educational project whose value lies in the *derivation*, not the score. With Stage 0
complete, we have an honest baseline (macro-F1 = 0.511) and a clean, leakage-free evaluation
protocol. With Stage 1 complete, we have a correct, gradient-checked implementation of an
embedding-based MLP — the Bengio et al. [1] architecture — written entirely in elementary
Octave matrix operations, trained end to end to a test macro-F1 of 0.608 (+9.7 pp over the
baseline). The most instructive empirical finding was that neither capacity nor the optimizer
was the limiting factor, but the strength of the class weighting: tempering inverse-frequency
weights ($\alpha = 0.5$) was worth more than every other intervention combined. This paper will
be updated as the project progresses through the curriculum.

---

## References

[1] Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin. *A Neural Probabilistic Language Model.*
Journal of Machine Learning Research, 3:1137–1155, 2003. — The fixed-context embedding-MLP
architecture this project implements (§2, §4.2).

[2] D. E. Rumelhart, G. E. Hinton, R. J. Williams. *Learning representations by back-propagating
errors.* Nature, 323:533–536, 1986. — Backpropagation (§4.5).

[3] V. Nair, G. E. Hinton. *Rectified Linear Units Improve Restricted Boltzmann Machines.* ICML,
2010. — ReLU activation (§2, §4.3).

[4] X. Glorot, A. Bordes, Y. Bengio. *Deep Sparse Rectifier Neural Networks.* AISTATS, 2011. —
ReLU in deep networks (§2).

[5] J. S. Bridle. *Probabilistic Interpretation of Feedforward Classification Network Outputs,
with Relationships to Statistical Pattern Recognition.* In Neurocomputing, Springer, 1990. —
Softmax output (§4.3).

[6] C. M. Bishop. *Pattern Recognition and Machine Learning.* Springer, 2006. — Softmax +
cross-entropy and its gradient (§4.4–§4.5).

[7] T. Mikolov, K. Chen, G. Corrado, J. Dean. *Efficient Estimation of Word Representations in
Vector Space.* ICLR Workshop, 2013. — Distributed word embeddings (§2).

[8] K. He, X. Zhang, S. Ren, J. Sun. *Delving Deep into Rectifiers: Surpassing Human-Level
Performance on ImageNet Classification.* ICCV, 2015. — He initialization for ReLU layers (§4.2,
§6).

[9] X. Glorot, Y. Bengio. *Understanding the difficulty of training deep feedforward neural
networks.* AISTATS, 2010. — Xavier/Glorot initialization (§2, baseline comparison).

[10] O. Tilk, T. Alumäe. *Bidirectional Recurrent Neural Network with Attention Mechanism for
Punctuation Restoration.* Interspeech, 2016. — Punctuation restoration as a neural task (§1.2, §2).

[11] D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. —
Considered but not needed; plain SGD sufficed (§4.7).

[12] *Wolne Lektury* — free Polish literary texts. Catalog: <https://wolnelektury.pl/katalog/>,
API: <https://wolnelektury.pl/api/>. — Corpus source (§3.1).

---

## Appendix A — Reproducibility

**Runtime.** GNU Octave (`octave-cli`), MATLAB-syntax compatible. No external ML dependencies.

**Data flow.** `data/raw/*.txt` → `src/preprocess.m` → `data/processed/{train,val,test}.mat`;
`src/train.m` → `data/processed/vocab.mat` (built from the train set if missing) and
`data/processed/model.mat` (trained weights). The `.mat` artifacts are committed to the
repository, so experiments reproduce without re-downloading or re-preprocessing the corpus.

**Run preprocessing and training.**

```bash
cd src
octave-cli preprocess.m   # writes ../data/processed/{train,val,test}.mat
octave-cli train.m        # writes ../data/processed/vocab.mat (if missing) + model.mat
```

**Continuous integration.** `.github/workflows/ci.yml` runs smoke tests for the data pipeline
(`tokenize`, `labelize`, `build_vocab`, `get_word_indices`, `build_windows`, `metrics`) and the
MLP components (`mlp_init`, `mlp_forward`, `mlp_loss`), plus the numerical gradient check
(`test_grad_check.m`). All tests must pass before merge.

**Source map.**

| Path | Role |
|------|------|
| `src/preprocess.m` | raw text → train/test split |
| `src/baseline_ngram.m` | Stage 0 bigram baseline |
| `src/mlp_init.m`, `mlp_forward.m`, `mlp_loss.m`, `mlp_backward.m` | Stage 1 MLP |
| `src/train.m` | Stage 1 training loop (SGD, early stopping on validation macro-F1) |
| `src/tests/` | smoke tests + numerical gradient check (`test_grad_check.m`) |
| `src/check.m` | standalone test-set evaluation (reproduces the reported macro-F1) |
| `src/detect.m` | interactive inference on arbitrary text (`<UNK>`-padded windows, no labels) |
| `src/lib/` | tokenize, labelize, process, build_vocab, get_word_indices, build_windows, metrics |
| `src/config/settings.m` | hyperparameters & data paths (`C_V`, `C_TRAIN_BOOKS`, …) |

**Project notes** (the source material for this paper):
[`learning-plan.md`](../notes/learning-plan.md) ·
[`stage-0-preprocess.md`](../notes/stage-0-preprocess.md) ·
[`stage-0-bigram-baseline.md`](../notes/stage-0-bigram-baseline.md) ·
[`stage-0-results.md`](../notes/stage-0-results.md) ·
[`stage-1-mlp.md`](../notes/stage-1-mlp.md).

---

## Appendix B — Notation and Key Formulas

A compact reference (cheat-sheet) for the symbols and equations used above.

### Notation

| Symbol | Meaning |
|--------|---------|
| $V$ | vocabulary size (5000) |
| $d$ | embedding dimension (50) |
| $h$ | hidden units (128) |
| $k$ | context radius (3 → window of $2k+1 = 7$ words) |
| $c$ | number of classes (3) |
| $N$ | batch size |
| $E$ | embedding matrix, $(V{+}1) \times d$ (last row = `<UNK>`) |
| $W_1, b_1$ | first linear layer ($h \times (2k{+}1)d$ and $h$) |
| $W_2, b_2$ | second linear layer ($c \times h$ and $c$) |
| $x$ | concatenated embeddings of the window, $\mathbb{R}^{(2k+1)d}$ |
| $s_1, a_1$ | pre-activation and ReLU activation of the hidden layer |
| $s_2, p$ | output logits and softmax probabilities |
| $y$ | one-hot target |
| $w_k$ | tempered inverse-frequency weight of class $k$: $(N_\text{total}/(c\,N_k))^{\alpha}$ |
| $\alpha$ | class-weight tempering exponent (0.5; 1 = full inverse frequency, 0 = uniform) |

### Key formulas (cross-referenced to the text)

| Quantity | Formula | Where |
|----------|---------|-------|
| Embedding concat | $x = [\,E_{w_{i-k}}; \dots; E_{w_{i+k}}\,]$ | §4.3, [stage-1-mlp.md](../notes/stage-1-mlp.md) |
| Hidden layer | $a_1 = \mathrm{ReLU}(W_1 x + b_1) = \max(0, W_1 x + b_1)$ | §4.3 |
| Softmax | $p_j = \dfrac{e^{s_{2,j}}}{\sum_l e^{s_{2,l}}}$ (subtract $\max$ for stability) | §4.3 |
| Weighted cross-entropy | $L = -\dfrac{1}{N}\sum_i w_{y_i}\,\log p_{i,y_i}$ | §4.4 |
| Output gradient | $\delta_2 = (p - y)\odot \dfrac{w}{N}$ | §4.5 |
| Hidden gradient | $\delta_1 = (\delta_2 W_2)\odot \mathbb{1}[s_1 > 0]$ | §4.5 |
| Weight gradients | $dW_2 = \delta_2^\top a_1,\quad dW_1 = \delta_1^\top x$ | §4.5 |
| Bias gradients | $db = \sum_i \delta_i$ | §4.5 |
| Embedding gradient | scatter-add: $dE(\text{idx},:)\mathrel{+}= \delta_\text{embed}$ | §4.5 |
| He initialization | $W \sim \mathcal{N}\!\big(0,\, 2/\text{fan-in}\big)$ | §4.2, [8] |
| Macro-F1 | $\tfrac{1}{c}\sum_{k} F1_k,\quad F1_k = \dfrac{2\,P_k R_k}{P_k + R_k}$ | §3.4, §5.1 |
| Gradient check | $\dfrac{|g_\text{analytic} - g_\text{num}|}{|g_\text{analytic}| + |g_\text{num}|} < 10^{-5}$ | §4.6 |
