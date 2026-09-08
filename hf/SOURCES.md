# Training data — sources and attribution

Every text in this corpus comes from [Wolne Lektury](https://wolnelektury.pl), a free Polish
digital library run by the Wolne Lektury Foundation. Each work is either in the public domain or
released under the [Free Art License 1.3](https://artlibre.org/licence/lal/pl/) (LAL 1.3) —
including the modern translations, which the foundation commissioned and published under that
licence.

The corpus is split by document, never by sentence, so no phrase leaks between train and test.

## Train (7 books, ~69% of tokens)

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Chłopi* | Władysław Stanisław Reymont | — | Public domain |
| *Lalka* | Bolesław Prus | — | Public domain |
| *Nad Niemnem* | Eliza Orzeszkowa | — | Public domain |
| *Proces* (Der Process) | Franz Kafka | Katarzyna Łakomik | LAL 1.3 |
| *Przedwiośnie* | Stefan Żeromski | — | Public domain |
| *Moralność pani Dulskiej* | Gabriela Zapolska | — | Public domain |
| *Mały Książę* (Le Petit Prince) | Antoine de Saint-Exupéry | Agata Kozak | LAL 1.3 |

## Validation (2 books, ~20% of tokens)

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Ziemia obiecana* | Władysław Stanisław Reymont | — | Public domain |
| *Rok 1984* (Nineteen Eighty-Four) | George Orwell | Julia Fiedorczuk | LAL 1.3 |

## Test (2 books, ~11% of tokens)

| Work | Author | Translator | Licence |
|---|---|---|---|
| *Syzyfowe prace* | Stefan Żeromski | — | Public domain |
| *Tajemniczy ogród* (The Secret Garden) | Frances Hodgson Burnett | Jadwiga Włodarkiewiczowa | Public domain |

## Notes

- *Chłopi* and *Lalka* are used as excerpts of the full novels; the rest are complete texts.
- The Wolne Lektury Foundation reserves the rights to its critical editions under art. 99(2) of the
  Polish Copyright Act. This corpus uses the plain text only.
- Terms of use: <https://wolnelektury.pl/info/zasady-wykorzystania/>
