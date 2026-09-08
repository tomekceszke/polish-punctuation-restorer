# Stage 0 — Preprocessing

See also: [learning-plan.md](learning-plan.md), [stage-0-bigram-baseline.md](stage-0-bigram-baseline.md)

## Goal

Transform raw `.txt` files into a numeric representation that `baseline_ngram.m` can consume: a list of tokens, their labels, and the document each token came from.

---

## Output of `preprocess.m`

Saved to two files after train/test split:

- `data/processed/train.mat` — variables: `train_words`, `train_labels`
- `data/processed/test.mat` — variables: `test_words`, `test_labels`

| Variable | Type | Description |
|---|---|---|
| `train_words` / `test_words` | cell array of strings | one token per position |
| `train_labels` / `test_labels` | numeric vector | 1=NONE, 2=COMMA, 3=PERIOD — label after each token |

All vectors within a file have the same length.

---

## Tokenisation

- Lowercase everything
- Remove all characters except `[a-ząćęłńóśźż ,.]` — keep Polish diacritics, keep comma and period
- Split on whitespace
- The punctuation character immediately after a token becomes its label; the character is then removed from the token stream

---

## Labels

Each token gets the label of the punctuation that follows it in the original text:

- nothing → NONE (1)
- `,` → COMMA (2)
- `.` → PERIOD (3)

---

## Document Tracking

Assign a numeric id to each source file (1 for the first book, 2 for the second, etc.). Every token gets the id of the file it came from. This is what allows a clean train/test split later — by document, not by position.

---

## Data Split

Split by document id, not by token position — to prevent phrase leakage between sets.

Target proportions: 90% train / 10% test. A document belongs entirely to one set — never split mid-book.

| Set | Books | Size | Share |
|---|---|---|---|
| **train** | chlopi, lalka, ziemia-obiecana, nad-niemnem, kafka-proces, przedwiosnie, moralnosc-pani-dulskiej, saint-exupery-maly-ksiaze, orwell-rok-1984 | ~7.98 MB | 90.1% |
| **test** | syzyfowe-prace, tajemniczy-ogrod | ~902 KB | 9.9% |

Test set rationale: `syzyfowe-prace` is original Polish prose (Żeromski — the same author appears in training via `przedwiosnie`); `tajemniczy-ogrod` is a translation, adding stylistic variety.

A document either belongs entirely to train or entirely to test. Never mix tokens from the same document across sets.

---

## Pitfalls

- **Last token of a document** — there is no next word. Assign NONE and handle the boundary explicitly, or discard the last token per document.
- **Abbreviations** — "ul.", "dr.", "nr." end with a period but are not sentence endings. At this stage it is acceptable to treat them as PERIOD and note it as a known limitation.
- **Multiple punctuation** — "...'" or "?!" — decide on a rule (take the first, take the last) and apply it consistently.

---

## Open item — ellipsis residue in the corpus (measured 2026-09-09, undecided)

`labelize.m` strips exactly **one** trailing mark, so an ellipsis written as `...` leaves the rest
of it glued to the word: `"co.."` labelled PERIOD, `"odmówią.."`, `"siedzi.."`. Measured while
exporting the corpus for Hugging Face: **16,204 of 828,125 train tokens (~2%)** still contain a
`,` or `.` inside the word string.

Consequences:

- `co` and `co..` are two different vocabulary entries competing for the same top-5000 slots.
- Almost every such token falls outside the vocabulary anyway and ends up as `<UNK>`, so the
  window carries less signal exactly where a sentence ends.
- The PERIOD class — already the weakest at F1 0.3811 — is the one polluted by it.

Options, none applied yet:

1. Strip **all** trailing `,` and `.` characters, label by the first one removed.
2. Collapse `...` to a single `.` during tokenisation, before labelling.
3. Leave it and treat it as corpus noise.

Any of the first two changes the training data, so it means retraining and a new Stage 1 number —
that is why nothing was touched. The published corpus and the published weights match the code as
it stands today; the artefact is documented in the Hugging Face dataset card. Decide this before
Stage 2, so the sequence model is not trained on the same noise.

