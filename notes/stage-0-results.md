# Stage 0 Results — Bigram Baseline

## Setup

- Train: Lalka, Quo Vadis, ... (80%)
- Test: Potop (10%)
- Vocab: top 1000 words from train only
- Model: bigram argmax (no weights)

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

## Notes

- NONE F1 high and stable (train ≈ test) — dominant class, easy to predict
- COMMA train vs test gap small — generalises reasonably
- PERIOD gap large (0.32 train vs 0.10 test) — model memorises PERIOD bigrams from training, fails on unseen ones
- Stage 1 MLP must beat Macro-F1 > 0.60