# Human-Like Anaphor Resolution in Large Language Models

This repository contains the materials needed to reproduce the analyses and figures for `cogsci_2026.pdf`.

## Data Download and License

The data and experiment materials are distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE-DATA.md` and the Creative Commons license text at https://creativecommons.org/licenses/by/4.0/.

Download options:

```bash
# Full repository, including code, paper, data, and generated outputs.
git clone https://github.com/wristy/anaphor.git

# GitHub source archive for the current main branch.
curl -L https://github.com/wristy/anaphor/archive/refs/heads/main.zip -o anaphor-main.zip

# Create a data-only archive with LICENSE-DATA.md and DATA_MANIFEST.md.
python scripts/make_data_release.py
```

The data-only archive is written to `dist/anaphor-cogsci-2026-data.zip` by default. It includes the source experiment materials, checked-in reference tables, generated tables/figures, and the data license/manifest. Model weights are not redistributed here; download them from Hugging Face under the license listed on each model card.

## Reproducing the Paper

Run the notebook:

```bash
jupyter notebook cogsci_2026_repro.ipynb
```

The notebook is the canonical reproduction entry point. By default it uses the checked-in paper reference data in `data/exp1.csv`, `data/exp2.csv`, and `data/exp3.csv`, rebuilds the surprisal tables, and writes figures to `generated/figures/`. It also rebuilds the comprehension-scoring summaries from the checked-in scored CSV files, writing summary CSVs to `data/` and PNG figures to `generated/figures/`. Surprisal figures use odd paper figure numbers (`fig1_surprisal`, `fig3_surprisal`, `fig5_surprisal`); comprehension figures use even paper figure numbers (`fig2_comprehension`, `fig4_comprehension`, `fig6_comprehension`).

To recompute model surprisals from the raw passage materials instead of using the checked-in data, set `DATA_SOURCE = 'regenerate'` in the notebook. This path is slower and requires local Hugging Face access plus enough memory for the selected models.

## Experiment Modules

The notebook and `paper_pipeline.py` keep the three repo experiments compartmentalized. The paper uses a different experiment order:

- Paper Experiment 1 -> repo `exp3`: title/topicality and distance manipulation, paper Figure 1.
- Paper Experiment 2 -> repo `exp1`: context/distance manipulation, paper Figure 3.
- Paper Experiment 3 -> repo `exp2`: semantic typicality/interference manipulation, paper Figure 5.

Run a subset by editing `EXPERIMENTS_TO_RUN` in `cogsci_2026_repro.ipynb`, for example:

```python
EXPERIMENTS_TO_RUN = ["exp3"]  # paper Experiment 1
```

Programmatic use:

```python
from pathlib import Path
from paper_pipeline import run_experiment

result = run_experiment(
    Path("."),
    "exp1",
    data_source="published",
)
result.wide_df.head()
```

## Hugging Face Models

The paper models are registered as aliases in `paper_pipeline.py`: `GPT2`, `Mistral-7B`, `pythia-12b-deduped`, `LLaMa-3.1-8B`, and `Mistral-24B`.

You can also run the three experiments with newer Hugging Face causal language models by passing repo ids directly:

```python
MODELS_TO_RUN = [
    "GPT2",
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B",
]
DATA_SOURCE = "regenerate"
```

For models that need a revision, custom dtype, or `trust_remote_code`, register an alias first:

```python
from paper_pipeline import register_hf_model

register_hf_model(
    "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B",
    torch_dtype="auto",
    display_name="Qwen2.5-1.5B",
)
MODELS_TO_RUN = ["GPT2", "Qwen2.5-1.5B"]
```

Private or gated Hugging Face models require local authentication before running regeneration:

```bash
hf auth login
```

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
- `data/`: checked-in paper reference surprisal tables and generated comprehension summary CSVs.
- `study 1 (context)/`, `study 2 (semantics)/`, `study3/`: source materials and human/answer data.
- `anaphor-automate-comprehesion-scoring/`: automated comprehension-scoring inputs and scored outputs used by the reproduction notebook.
- `generated/`: reproducible output tables and PNG figures created by the notebook.

## Citation

If you use these materials, please cite the paper:

```bibtex
@inproceedings{zhang2026anaphor,
  title = {Human-Like Anaphor Resolution in Large Language Models},
  author = {Zhang, Keane and Chinta, Varshini and Shah, Raj Sanjay and Varma, Sashank},
  booktitle = {Proceedings of the Annual Meeting of the Cognitive Science Society},
  year = {2026},
  note = {CogSci 2026 submission}
}
```
