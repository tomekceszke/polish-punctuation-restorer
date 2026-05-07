# Stage 0 Results — Bigram Baseline

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

## Per-Class Metrics (V=5000)

| Class  | Train F1 | Test F1 |
|--------|----------|---------|
| NONE   | 0.9403   | 0.9255  |
| COMMA  | 0.6492   | 0.4599  |
| PERIOD | 0.5580   | 0.1463  |
| Macro  | 0.7159   | **0.5106** |

**Stage 1 MLP must beat Macro-F1 > 0.611**

---

## Archive: V=1000 Results

## Confusion Matrix

```
              Predicted
              NONE   COMMA  PERIOD
Actual NONE  [105608   878    441]
       COMMA [  8179  4764    190]
       PERIOD[  6939   762    473]
```

## Per-Class Metrics

| Class  | Precision | Recall | F1     |
|--------|-----------|--------|--------|
| NONE   | 0.8748    | 0.9877 | 0.9278 |
| COMMA  | 0.7439    | 0.3628 | 0.4877 |
| PERIOD | 0.4284    | 0.0579 | 0.1020 |

## Summary

| Metric    | Value  |
|-----------|--------|
| Macro-F1  | 0.5058 |

## Train vs Test Comparison

| Class  | Train F1 | Test F1 |
|--------|----------|---------|
| NONE   | 0.9220   | 0.9278  |
| COMMA  | 0.5114   | 0.4877  |
| PERIOD | 0.3187   | 0.1020  |
| Macro  | 0.5840   | 0.5058  |

## Notes (V=1000)

- NONE F1 high and stable (train ≈ test) — dominant class, easy to predict
- COMMA train vs test gap small — generalises reasonably
- PERIOD gap large (0.32 train vs 0.10 test) — model memorises PERIOD bigrams from training, fails on unseen ones