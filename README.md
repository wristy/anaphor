# Human-Like Anaphor Resolution in Large Language Models

This repository contains the materials needed to reproduce the analyses and figures for `cogsci_2026.pdf`.

## Reproducing the Paper

Run the notebook:

```bash
jupyter notebook cogsci_2026_repro.ipynb
```

The notebook is the canonical reproduction entry point. By default it uses the checked-in paper reference data in `data/exp1.csv`, `data/exp2.csv`, and `data/exp3.csv`, rebuilds the surprisal tables, and writes figures to `generated/figures/`. It also rebuilds the comprehension-scoring summaries from the checked-in scored CSV files.

To recompute model surprisals from the raw passage materials instead of using the checked-in data, set `DATA_SOURCE = 'regenerate'` in the notebook. This path is slower and requires local Hugging Face access plus enough memory for the selected models.

## Setup

Create a fresh Python environment and install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name anaphor-cogsci-2026
```

The default notebook path does not require API keys. Optional comprehension rescoring uses Gemini through `google-generativeai`; set `GOOGLE_API_KEY` and `RUN_COMPREHENSION_RESCORING = True` in the notebook only if you want to rerun that judge model.

## Repository Map

- `cogsci_2026.pdf`: paper to reproduce.
- `cogsci_2026_repro.ipynb`: reproduction notebook for the paper analyses and figures.
- `paper_pipeline.py`: surprisal data loading, regeneration, validation, and plotting helpers.
- `comprehension_pipeline.py`: comprehension-scoring loading, summarization, optional rescoring, and plotting helpers.
- `data/`: checked-in paper reference surprisal tables.
- `study 1 (context)/`, `study 2 (semantics)/`, `study3/`: source materials and human/answer data.
- `anaphor-automate-comprehesion-scoring/`: automated comprehension-scoring inputs and scored outputs used by the reproduction notebook.
- `generated/`: reproducible output tables and figures created by the notebook.

