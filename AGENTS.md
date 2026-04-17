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

## Vehicles Counter (`vehicles-counter/`)
- **Requirements**: Octave, VLC (for live stream). Standalone project with its own git VCS.
- **Config**: `conf/settings.m` — edit `hidden_layer_size`, `lambda`, `iterations`, `resize_scale`, `datasource_path_prefix` before running.
- **Core scripts** (run from `vehicles-counter/`):
  - `learn.m` — trains NN. Reads `datasource/train/{cars,notcars}/`. Saves `Theta1.mat`, `Theta2.mat`.
  - `check.m` — evals on `datasource/test/{cars,notcars}/`. Must have pre-trained `Theta1.mat`/`Theta2.mat`.
  - `detect.m` — live vehicle count via VLC stream.
- **Optimizer**: `lib/fmincg.m` (conjugate gradients, from Carl Edward Rasmussen). Not Octave built-in — must stay in `lib/`.
- **Windows `.bat` files** in `bin/` — require porting to sh on macOS/Linux.
- **Gotcha**: `learn.m` and `check.m` must be run in the Octave working directory matching `conf/settings.m` paths (relative). Running from elsewhere breaks data source resolution.

## Learning Plan
- `plan-nauki.md` — 5-stage curriculum (ngram baseline → MLP with hand-written backprop → bi-LSTM → transformer → deploy) for the Polish Punctuation Restorer project. Contains architecture, hyperparams, mathematical derivations needed, and trap avoidance.
