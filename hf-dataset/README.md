---
language:
  - pl
license: other
license_name: public-domain-and-free-art-license-1.3
license_link: https://artlibre.org/licence/lal/pl/
task_categories:
  - token-classification
size_categories:
  - 1M<n<10M
tags:
  - punctuation-restoration
  - polish
  - literature
  - wolne-lektury
pretty_name: Polish Punctuation Corpus (Wolne Lektury)
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.parquet
      - split: validation
        path: data/val.parquet
      - split: test
        path: data/test.parquet
---

# Polish Punctuation Corpus (Wolne Lektury)

Word-level punctuation labels extracted from 11 Polish literary works, built to train
[tomekceszke/polish-punctuation-restorer](https://huggingface.co/tomekceszke/polish-punctuation-restorer).

Each row is one word and the mark that followed it in the original text.

| Column | Type | Meaning |
|---|---|---|
| `word` | string | Lower-cased token, punctuation removed |
| `label` | int | `1` = nothing follows, `2` = comma, `3` = period |

```python
from datasets import load_dataset

ds = load_dataset("tomekceszke/polish-punctuation-corpus")
ds["train"][0]        # {'word': 'chłopi', 'label': 1}
```

## Splits

The split is **by document, never by sentence** — no phrase can leak from training into
evaluation.

| Split | Rows | Share | Books |
|---|---|---|---|
| train | 828,125 | 69.2% | 7 |
| validation | 239,580 | 20.0% | 2 |
| test | 128,235 | 10.7% | 2 |

## Label distribution

Heavily imbalanced, which is the central difficulty of the task: four out of five words are
followed by nothing at all.

| Split | NONE (1) | COMMA (2) | PERIOD (3) |
|---|---|---|---|
| train | 668,692 (80.7%) | 98,756 (11.9%) | 60,677 (7.3%) |
| validation | 191,618 (80.0%) | 30,825 (12.9%) | 17,137 (7.2%) |
| test | 106,928 (83.4%) | 13,133 (10.2%) | 8,174 (6.4%) |

Any model trained on this without class weighting will learn to predict `1` and stop there.

## How it was built

Raw `.txt` files from [Wolne Lektury](https://wolnelektury.pl) go through
[`src/preprocess.m`](https://github.com/tomekceszke/polish-punctuation-restorer/blob/main/src/preprocess.m):

1. Lower-case the whole text.
2. Strip every character except Polish letters (`a-ząćęłńóśźż`), whitespace, `,` and `.`.
3. Split on whitespace.
4. Take one trailing `,` or `.` off each token; that mark becomes the label.

Only commas and periods are modelled. Question marks, exclamation marks and semicolons are
removed in step 2 and become `NONE`.

### Known artefact

Step 4 removes exactly **one** trailing mark, so an ellipsis (`…` written as `...`) leaves a
residue: `"co.."` labelled as `PERIOD`. About 2% of tokens carry such a leftover character. The
corpus is published as it is, because this is the exact data the released model was trained on.

## Sources and licences

Every text comes from Wolne Lektury and is either in the public domain or released under the
[Free Art License 1.3](https://artlibre.org/licence/lal/pl/) — the modern translations included,
which the foundation commissioned and published under that licence.

### Train

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Chłopi* | Władysław Stanisław Reymont | — | Public domain |
| *Lalka* | Bolesław Prus | — | Public domain |
| *Nad Niemnem* | Eliza Orzeszkowa | — | Public domain |
| *Proces* | Franz Kafka | Katarzyna Łakomik | Free Art 1.3 |
| *Przedwiośnie* | Stefan Żeromski | — | Public domain |
| *Moralność pani Dulskiej* | Gabriela Zapolska | — | Public domain |
| *Mały Książę* | Antoine de Saint-Exupéry | Agata Kozak | Free Art 1.3 |

### Validation

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Ziemia obiecana* | Władysław Stanisław Reymont | — | Public domain |
| *Rok 1984* | George Orwell | Julia Fiedorczuk | Free Art 1.3 |

### Test

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Syzyfowe prace* | Stefan Żeromski | — | Public domain |
| *Tajemniczy ogród* | Frances Hodgson Burnett | Jadwiga Włodarkiewiczowa | Public domain |

*Chłopi* and *Lalka* are used as excerpts; the rest are complete texts. The Wolne Lektury
Foundation reserves the rights to its critical editions under art. 99(2) of the Polish Copyright
Act — this corpus uses the plain text only. Terms of use:
<https://wolnelektury.pl/info/zasady-wykorzystania/>

## Links

- **Model trained on this:** [tomekceszke/polish-punctuation-restorer](https://huggingface.co/tomekceszke/polish-punctuation-restorer)
- **Code:** <https://github.com/tomekceszke/polish-punctuation-restorer>
- **Project page:** <https://tomek.ceszke.com/polish-punctuation-restorer/>
