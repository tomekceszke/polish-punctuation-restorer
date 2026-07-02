# Paper

[`paper.md`](paper.md) is the academic write-up of the Polish Punctuation Restorer (PPR)
project — goal, assumptions, methodology, results, and conclusions. It is a
**work in progress**: Stage 0 (bigram baseline, macro-F1 0.511) and Stage 1 (the MLP,
macro-F1 0.608) are reported with final numbers; Stages 2–5 are outlook. The document
will be revised as the project advances.

The source of truth is the repository [`README.md`](../README.md) and the design notes in
[`../notes/`](../notes/); the paper consolidates and formalizes them.

## Building the PDF (deferred)

The paper is written in pandoc-flavored Markdown (LaTeX math via `$…$` / `$$…$$`), so it
renders to PDF without edits. No toolchain is installed yet; to build:

```bash
# Prerequisites (macOS)
brew install pandoc
brew install tectonic        # lightweight LaTeX engine (downloads packages on demand)

# Build
cd paper
pandoc paper.md -o paper.pdf --pdf-engine=tectonic -N --toc
```

`-N` numbers sections; `--toc` adds a table of contents. Run the command from inside
`paper/` so the relative links to `../notes/` and `../README.md` resolve.
