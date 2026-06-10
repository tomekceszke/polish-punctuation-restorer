# Stage 0 Results — Bigram Baseline

These results are summarized in the project paper, §5.1: [`../paper/paper.md`](../paper/paper.md).

## Setup (V=5000 — current)

- Train: 9 books (Chłopi, Lalka, Ziemia Obiecana, Nad Niemnem, Proces, Przedwiośnie, Moralność Pani Dulskiej, Mały Książę, 1984) — 1,067,705 tokens
- Test: 2 books (Syzyfowe Prace, Tajemniczy Ogród) — 128,235 tokens
- Vocab: top 5000 words from train only
- Model: bigram argmax (no weights)

## Confusion Matrix (V=5000)

```
              Predicted
              NONE   COMMA  PERIOD
Actual NONE  [105082  1114    731]
       COMMA [  8356  4473    304]
       PERIOD[  6716   731    727]
```

Matrix totals 128,234 — one less than the 128,235 test tokens: the last token of the corpus has no following word, so the bigram model drops it.

## Per-Class Metrics (V=5000)

| Class  | Precision (test) | Recall (test) | Test F1 | Train F1 |
|--------|------------------|---------------|---------|----------|
| NONE   | 0.875 | 0.983 | 0.9255 | 0.9403 |
| COMMA  | 0.708 | 0.341 | 0.4599 | 0.6492 |
| PERIOD | 0.413 | 0.089 | 0.1463 | 0.5580 |
| Macro  | —     | —     | **0.5106** | 0.7159 |

Accuracy (context only, not the primary metric): **0.860** — inflated by the dominant NONE class.

**Stage 1 MLP must beat Macro-F1 > 0.611**

## Expected vs Actual

The plan predicted COMMA F1 0.15–0.30 and PERIOD F1 0.40–0.55 — the opposite of what happened on test (COMMA 0.460, PERIOD 0.146). On train the prediction roughly held (PERIOD 0.558); the flip is a test-time generalization failure: the model memorises period-ending bigrams and rarely sees them again in unseen books, while comma contexts (e.g. before conjunctions) repeat across books.

---

## Archive: V=1000 Results

### Confusion Matrix

```
              Predicted
              NONE   COMMA  PERIOD
Actual NONE  [105608   878    441]
       COMMA [  8179  4764    190]
       PERIOD[  6939   762    473]
```

### Per-Class Metrics

| Class  | Precision | Recall | F1     |
|--------|-----------|--------|--------|
| NONE   | 0.8748    | 0.9877 | 0.9278 |
| COMMA  | 0.7439    | 0.3628 | 0.4877 |
| PERIOD | 0.4284    | 0.0579 | 0.1020 |

### Summary

| Metric    | Value  |
|-----------|--------|
| Macro-F1  | 0.5058 |

### Train vs Test Comparison

| Class  | Train F1 | Test F1 |
|--------|----------|---------|
| NONE   | 0.9220   | 0.9278  |
| COMMA  | 0.5114   | 0.4877  |
| PERIOD | 0.3187   | 0.1020  |
| Macro  | 0.5840   | 0.5058  |

### Notes (V=1000)

- NONE F1 high and stable (train ≈ test) — dominant class, easy to predict
- COMMA train vs test gap small — generalises reasonably
- PERIOD gap large (0.32 train vs 0.10 test) — model memorises PERIOD bigrams from training, fails on unseen ones