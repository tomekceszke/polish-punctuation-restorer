#!/usr/bin/env bash
#   Polish Punctuation Restorer
#   Author: Tomasz Ceszke 2026
#
#   Assembles hf/ — the staging directory uploaded to the Hugging Face model repo.
#   Hand-written files (README.md, SOURCES.md) are never touched; everything else is regenerated.
#
#   Run from src/:  ./utils/build_hf.sh

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$SRC")"
HF="$ROOT/hf"

#   weights in portable formats (model_v7.mat + vocab.txt)
cd "$SRC"
octave-cli utils/export_hf.m

#   originals, so the Octave path works without the rest of the repo
cp "$ROOT/data/processed/model.mat" "$HF/model.mat"
cp "$ROOT/data/processed/vocab.mat" "$HF/vocab.mat"
cp "$ROOT/LICENSE" "$HF/LICENSE"

#   minimal inference bundle — no training code
mkdir -p "$HF/inference/lib" "$HF/inference/config"
cp "$SRC/mlp_forward.m" "$HF/inference/mlp_forward.m"
cp "$SRC/config/settings.m" "$HF/inference/config/settings.m"
for f in tokenize labelize get_word_indices build_windows post_process; do
    cp "$SRC/lib/$f.m" "$HF/inference/lib/$f.m"
done

#   detect.m loads the weights one level up in the bundle, not from data/processed/
sed -e "s|'\.\./data/processed/model\.mat'|'../model.mat'|" \
    -e "s|'\.\./data/processed/vocab\.mat'|'../vocab.mat'|" \
    "$SRC/detect.m" > "$HF/inference/detect.m"

echo "hf/ assembled at $HF"
