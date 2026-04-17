# Agents Guide

## Cross-cutting
- **Runtime**: GNU Octave (`octave-cli` recommended). MATLAB syntax compatible, tested with Octave.
- **`vehicles-counter`** is a separate git repo — ignored from root via `.gitignore`. Work in it as its own project.
- **`.mat` files are committed** — pre-computed weights and labels are stored in the repo for reproducibility.
- **VS Code**: run/debug `.m` files via the **Octave Debugger** extension. Config in `.vscode/launch.json` (runs `${file}` with `octave-cli`).

## Text Preprocessing (`src/`)
- **Entrypoint**: `src/preprocess.m`
- **Lib**: `src/lib/tokenize.m`, `src/lib/labelize.m` (added via `addpath('lib')`).
- **Run**: `octave-cli preprocess.m` (from `src/`).
- **Data Flow**: `data/raw/*.txt` → `preprocess.m` → `data/processed/data.mat` (exports `words`, `labels`).
- **Gotcha**: `preprocess.m` has hardcoded input paths (`lalka.txt`, `chlopi.txt`). Edit the `books` cell array to change sources.

## Vehicles Counter
- `vehicles-counter/` is a separate project temporarily stored here for convenience. It will be removed soon.


## Learning Plan
- `plan-nauki.md` — 5-stage curriculum (ngram baseline → MLP with hand-written backprop → bi-LSTM → transformer → deploy) for the Polish Punctuation Restorer project. Contains architecture, hyperparams, mathematical derivations needed, and trap avoidance.
