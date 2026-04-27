# Stage 0 — Preprocessing

See also: [learning-plan.md](learning-plan.md), [stage-0-bigram-baseline.md](stage-0-bigram-baseline.md)

## Goal

Transform raw `.txt` files into a numeric representation that `baseline_ngram.m` can consume: a list of tokens, their labels, and the document each token came from.

---

## Output of `preprocess.m`

Saved to `data/processed/data.mat`:

| Variable | Type | Description |
|---|---|---|
| `words` | cell array of strings | one token per position |
| `labels` | numeric vector | 1=NONE, 2=COMMA, 3=PERIOD — label after each token |
| `doc_ids` | numeric vector | which document each token came from (1, 2, 3, ...) |

All three vectors have the same length.

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

| Set | Dzieła | Rozmiar | Udział |
|---|---|---|---|
| **train** | chlopi, lalka, ziemia-obiecana, nad-niemnem, kafka-proces, przedwiosnie, moralnosc-pani-dulskiej, saint-exupery-maly-ksiaze, orwell-rok-1984 | ~7.98 MB | 90.1% |
| **test** | syzyfowe-prace, tajemniczy-ogrod | ~902 KB | 9.9% |

Uzasadnienie testu: `syzyfowe-prace` to proza oryginalna (Żeromski — autor jest też w zbiorze treningowym przez `przedwiosnie`), `tajemniczy-ogrod` to tłumaczenie — wnosi różnorodność stylistyczną.

A document either belongs entirely to train or entirely to test. Never mix tokens from the same document across sets.

---

## Pitfalls

- **Last token of a document** — there is no next word. Assign NONE and handle the boundary explicitly, or discard the last token per document.
- **Abbreviations** — "ul.", "dr.", "nr." end with a period but are not sentence endings. At this stage it is acceptable to treat them as PERIOD and note it as a known limitation.
- **Multiple punctuation** — "...'" or "?!" — decide on a rule (take the first, take the last) and apply it consistently.
