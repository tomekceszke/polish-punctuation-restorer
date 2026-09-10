# Web demo — running the model in the browser

The landing page at [tomek.ceszke.com/polish-punctuation-restorer](https://tomek.ceszke.com/polish-punctuation-restorer/)
runs the real Stage 1 model. GitHub Pages serves static files only, so there is no backend: the
weights are downloaded once and the forward pass runs in JavaScript, in the visitor's browser.

This is cheap because the model is small — 295,365 parameters, 1.18 MB as float32, and about 45K
multiply-adds per word. Measured in headless Chrome: **2,100 words in ~104 ms**.

## The two halves of the demo box

The page used to show a scripted animation: type a sentence, press an imaginary button, pop the marks
in one by one. That animation explains the project to a first-time visitor in about eight seconds and
is worth keeping, so the real input did not get its own separate widget.

Instead there is **one box with two states**:

- **attract** — the animation loops, exactly as before. The button is inert (`pointer-events: none`)
  and presses itself.
- **live** — the first click, keystroke or focus tears the loop down (the `generation` counter),
  empties the box, makes it `contenteditable` and hands the button over to the visitor. If the box is
  left empty and untouched for 20 s, the animation comes back.

The result of a real run is rendered through the *same* `renderTokens()` + mark-reveal animation that
the scripted demo uses. The visitor is shown a promise, then the identical thing actually happening.

## Weight export

`src/utils/export_web.m` writes `web/model/`:

| File | What |
|---|---|
| `weights.bin` | Five tensors back to back, row-major float32, little-endian — 1,181,460 bytes |
| `vocab.txt` | 5000 words, one per line; line number = model index, `<UNK>` = 5001 |
| `meta.json` | Shapes, byte offsets, hyperparameters, and a self-test vector |

Tensor order and offsets:

| Tensor | Shape | Floats | Byte offset |
|---|---|---|---|
| `E` | 5001 × 50 | 250,050 | 0 |
| `W1` | 128 × 350 | 44,800 | 1,000,200 |
| `b1` | 128 | 128 | 1,179,400 |
| `W2` | 3 × 128 | 384 | 1,179,912 |
| `b2` | 3 | 3 | 1,181,448 |

Octave stores matrices column-major, so each one is transposed before `fwrite` — `fwrite(fid,
single(X.'), 'single')` lays `X` out row by row, which is what the JavaScript loader expects. The
exporter refuses to write anything whose shape disagrees with `config/settings.m`, and verifies the
file length afterwards.

`web/model.js` fetches the blob once and creates `Float32Array` **views** into it at those offsets —
no copying, no parsing. Cache-busting comes from the `ppr-build` meta tag, which the Pages workflow
rewrites to the commit SHA; `model.js` appends it to every `model/` request, so a retrained model is
never served from a stale cache.

**After every retraining:** `octave-cli utils/export_web.m`, then commit `web/model/`.

## Parity with `detect.m`

`meta.json` carries a `selfTest` block: one sentence pushed through the real Octave pipeline at export
time, together with its exact output. `PPR.selfTest()` replays it in the browser on first load and
warns in the console on any mismatch. It is regenerated on every export, so it cannot go stale.

Two places where the JavaScript deliberately does **not** copy Octave:

1. **Capitalization of non-ASCII letters.** `post_process.m:17` guards the uppercase call with
   `txt(i) < 128`, so a sentence starting with `ósmy` comes out lowercase — asserted as-is in
   `src/tests/test_post_process.m:43-44`. JavaScript's `toUpperCase()` handles `ó` correctly and is
   allowed to. Worth fixing on the Octave side eventually.
2. **Empty words.** `labelize.m` splits one trailing mark off a token, so a standalone `,` becomes an
   empty-string word. The JS tokenizer drops those. This only shows up on malformed input.

Everything else is a faithful port, including the details that matter:

- `lower()` and the `[^a-ząćęłńóśźż\s,.]` strip are UTF-8-aware in Octave 11, so the same JavaScript
  regex gives the same tokens — digits, hyphens and apostrophes are dropped, and `e-mail` becomes
  `email`.
- Any punctuation the visitor types is thrown away; the model re-predicts from scratch.
- The index vector is padded with `k` `<UNK>` entries at both ends, so the last word also gets a
  window — that is where the closing period comes from.
- `argmax` breaks ties toward the lowest class index, matching Octave's `max`.

### The empty vocabulary slot

`vocab.txt` line 117 is empty, and that is not a bug in the export. The empty-word artefact from
`labelize.m` was frequent enough in the corpus to land in the top 5000 — a standalone comma appears
that often in literary dialogue. The row has to stay so every later index keeps its place; nothing in
the browser ever maps to it, since the JS tokenizer drops empty words.

## Verifying a change

```bash
cd src
octave-cli utils/export_web.m        # shapes + byte count must match the table above
printf 'napisz kiedy dojedziesz na miejsce bo zaczynam się martwić\nexit\n' | octave-cli detect.m

cd ../web && python3 -m http.server 8765
```

Then open the page, type the same sentence and compare character by character, and check the console
for a self-test warning.
