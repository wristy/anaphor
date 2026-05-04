from pathlib import Path

import pandas as pd

from paper_pipeline import (
    EXPERIMENTS,
    generate_experiment_wide,
    plot_surprisal_figure,
    save_experiment_outputs,
)


ROOT = Path(__file__).resolve().parent
GENERATED_DATA_DIR = ROOT / "generated" / "data"
GENERATED_FIGURE_DIR = ROOT / "generated" / "figures"

MODEL = "LLaMa-3.1-8B"

for experiment in ["exp1", "exp2", "exp3"]:
    print(f"=== {experiment}: re-scoring {MODEL} ===", flush=True)
    existing = pd.read_csv(GENERATED_DATA_DIR / f"{experiment}.csv")
    fresh = generate_experiment_wide(ROOT, experiment, [MODEL])

    if len(fresh) != len(existing):
        raise RuntimeError(
            f"{experiment}: row count mismatch — existing {len(existing)}, fresh {len(fresh)}"
        )

    existing[MODEL] = fresh[MODEL].values
    save_experiment_outputs(ROOT, experiment, existing, GENERATED_DATA_DIR)
    plot_surprisal_figure(experiment, existing, GENERATED_FIGURE_DIR)
    print(existing.head(), flush=True)
