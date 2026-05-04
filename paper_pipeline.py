from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "GPT2",
    "Mistral-7B",
    "pythia-12b-deduped",
    "LLaMa-3.1-8B",
    "Mistral-24B",
]

PAPER_MODEL_ORDER = [
    "GPT2",
    "LLaMa-3.1-8B",
    "pythia-12b-deduped",
    "Mistral-7B",
    "Mistral-24B",
]

MODEL_DISPLAY_NAMES = {
    "GPT2": "GPT2-XL",
    "Mistral-7B": "Mistral-7B",
    "pythia-12b-deduped": "Pythia-12B",
    "LLaMa-3.1-8B": "LLaMa3.1-8B",
    "Mistral-24B": "Mistral-24B",
}


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    architecture: str
    revision: str | None = None
    torch_dtype: str = "float32"


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    title: str
    figure_number: int
    passage_ids: list[int]
    legacy_version_order: list[str]
    paper_version_order: list[str]
    legend_labels: dict[str, str]
    version_hatches: dict[str, str]
    legend_fontsize: int
    data_builder: Callable[[Path], list[dict[str, object]]]


MODEL_SPECS = {
    "GPT2": ModelSpec("openai-community/gpt2-xl", "causal", torch_dtype="float32"),
    "Mistral-7B": ModelSpec(
        "mistralai/Mistral-7B-v0.1", "causal", torch_dtype="float16"
    ),
    "pythia-12b-deduped": ModelSpec(
        "EleutherAI/pythia-12b-deduped",
        "causal",
        revision="step143000",
        torch_dtype="float32",
    ),
    "LLaMa-3.1-8B": ModelSpec(
        "meta-llama/Llama-3.1-8B", "causal", torch_dtype="float32"
    ),
    "Mistral-24B": ModelSpec(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "causal",
        torch_dtype="float16",
    ),
}


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _strip_light_punctuation(text: str) -> str:
    return (
        text.replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
    )


def _clean_words(text: str) -> list[str]:
    return _strip_light_punctuation(_collapse_whitespace(text)).split()


def _sem(values: pd.Series) -> float:
    numeric = values.dropna().astype(float)
    if len(numeric) <= 1:
        return np.nan
    return float(numeric.std(ddof=1) / math.sqrt(len(numeric)))


def _read_exp1_answer_words(root: Path) -> dict[int, str]:
    path = root / "study 1 (context)" / "1 materials" / "scoring" / "question-scoring.xlsx"
    df = pd.read_excel(path)
    antecedents = (
        df[(df["Version"] == "A") & (df["Question"] == 10) & (df["Passage"] != 4)]
        .sort_values("Passage")
        .set_index("Passage")["Answer"]
    )
    return {int(passage): str(answer).strip() for passage, answer in antecedents.items()}


def _read_exp2_answer_words(root: Path) -> dict[int, str]:
    path = root / "study 2 (semantics)" / "2 materials" / "scoring" / "quest-scoring.xlsx"
    df = pd.read_excel(path)
    antecedents = (
        df[(df["Version"] == "A") & (df["Question"] == 10) & (df["Passage"] != 4)]
        .sort_values("Passage")
        .set_index("Passage")["Answer"]
    )
    return {int(passage): str(answer).strip() for passage, answer in antecedents.items()}


def _read_exp3_target_words(root: Path) -> dict[int, dict[str, str]]:
    path = root / "filled_words.csv"
    df = pd.read_csv(path)
    mapping: dict[int, dict[str, str]] = {}
    for passage_id, row in enumerate(df.itertuples(index=False), start=1):
        mapping[passage_id] = {
            "anaphor_1": str(row.anaphor_1).strip(),
            "anaphor_2": str(row.anaphor_2).strip(),
        }
    return mapping


def _predecessor_word(text: str, target_word: str) -> str:
    words = _clean_words(text)
    index = words.index(target_word)
    if index == 0:
        raise ValueError(f"Target word {target_word!r} has no predecessor in text")
    return words[index - 1]


def _last_occurrence_surprisal_index(text: str, cue_word: str) -> int:
    words = _clean_words(text)
    return len(words) - words[::-1].index(cue_word) - 1


def build_exp1_items(root: Path) -> list[dict[str, object]]:
    answer_words = _read_exp1_answer_words(root)
    items: list[dict[str, object]] = []
    for passage_id in [1, 2, 3, *range(5, 21)]:
        cue_text = (
            root
            / "study1"
            / "version_a"
            / f"p{passage_id:02d}A.txt"
        ).read_text()
        cue_word = _predecessor_word(cue_text, answer_words[passage_id])
        for version in ["A", "B", "C", "D"]:
            text = (
                root
                / "study1"
                / f"version_{version.lower()}"
                / f"p{passage_id:02d}{version}.txt"
            ).read_text()
            items.append(
                {
                    "passage_id": passage_id,
                    "version": version,
                    "text": _collapse_whitespace(text),
                    "cue_word": cue_word,
                }
            )
    return items


def build_exp2_items(root: Path) -> list[dict[str, object]]:
    answer_words = _read_exp2_answer_words(root)
    items: list[dict[str, object]] = []
    for passage_id in [1, 2, 3, *range(5, 21)]:
        cue_text = (
            root
            / "study2"
            / "version_a"
            / f"p{passage_id:02d}a.txt"
        ).read_text()
        cue_word = _predecessor_word(cue_text, answer_words[passage_id])
        for version in ["A", "B", "C", "D"]:
            suffix = version.lower()
            text = (
                root
                / "study2"
                / f"version_{suffix}"
                / f"p{passage_id:02d}{suffix}.txt"
            ).read_text()
            items.append(
                {
                    "passage_id": passage_id,
                    "version": version,
                    "text": _collapse_whitespace(text),
                    "cue_word": cue_word,
                }
            )
    return items


def _build_exp3_versions(raw_text: str) -> dict[str, str]:
    lines = raw_text.splitlines()
    modified = [line.replace("1", lines[2]).replace("2", lines[3]) for line in lines[1:]]

    title_1 = modified[0].strip()
    title_2 = modified[1].strip()
    anaphor_1 = modified[4].strip()
    anaphor_2 = modified[5].strip()
    body = modified[10:-1]

    return {
        "A": _collapse_whitespace(" ".join([title_2, *body, anaphor_2])),
        "C": _collapse_whitespace(" ".join([title_1, *body, anaphor_2])),
        "B": _collapse_whitespace(" ".join([title_1, *body, anaphor_1])),
        "D": _collapse_whitespace(" ".join([title_2, *body, anaphor_1])),
    }


def build_exp3_items(root: Path) -> list[dict[str, object]]:
    target_words = _read_exp3_target_words(root)
    items: list[dict[str, object]] = []
    for passage_id in range(1, 17):
        raw_text = (root / "study3" / f"{passage_id}.txt").read_text()
        versions = _build_exp3_versions(raw_text)
        raw_lines = raw_text.splitlines()
        modified = [line.replace("1", raw_lines[2]).replace("2", raw_lines[3]) for line in raw_lines[1:]]
        anaphor_text_by_version = {
            "A": modified[5].strip(),
            "B": modified[4].strip(),
            "C": modified[5].strip(),
            "D": modified[4].strip(),
        }
        target_word_by_version = {
            "A": target_words[passage_id]["anaphor_2"],
            "B": target_words[passage_id]["anaphor_1"],
            "C": target_words[passage_id]["anaphor_2"],
            "D": target_words[passage_id]["anaphor_1"],
        }
        for version in ["A", "C", "B", "D"]:
            items.append(
                {
                    "passage_id": passage_id,
                    "version": version,
                    "text": versions[version],
                    "anaphor_text": anaphor_text_by_version[version],
                    "target_word": target_word_by_version[version],
                }
            )
    return items


EXPERIMENTS = {
    "exp1": ExperimentSpec(
        slug="exp1",
        title="Experiment 1",
        figure_number=1,
        passage_ids=[1, 2, 3, *range(5, 21)],
        legacy_version_order=["A", "B", "C", "D"],
        paper_version_order=["A", "C", "B", "D"],
        legend_labels={
            "A": "A: near spat. dist., short temp. dur.",
            "C": "B: near spat. dist, long temp. dur.",
            "B": "C: far spat. dist., short temp. dur.",
            "D": "D: far spat. dist., long temp. dur.",
        },
        version_hatches={"A": "", "C": "///", "B": "XX", "D": "\\\\"},
        legend_fontsize=15,
        data_builder=build_exp1_items,
    ),
    "exp2": ExperimentSpec(
        slug="exp2",
        title="Experiment 2",
        figure_number=3,
        passage_ids=[1, 2, 3, *range(5, 21)],
        legacy_version_order=["A", "B", "C", "D"],
        paper_version_order=["B", "A", "D", "C"],
        legend_labels={
            "B": "A: typ. ante., atyp. distractor",
            "A": "B: typ. ante., typ. distractor",
            "D": "C: atyp. ante., atyp. distractor",
            "C": "D: atyp. ante., typ. distractor",
        },
        version_hatches={"B": "", "A": "///", "D": "xx", "C": "\\\\"},
        legend_fontsize=16,
        data_builder=build_exp2_items,
    ),
    "exp3": ExperimentSpec(
        slug="exp3",
        title="Experiment 3",
        figure_number=5,
        passage_ids=list(range(1, 17)),
        legacy_version_order=["A", "C", "B", "D"],
        paper_version_order=["A", "C", "B", "D"],
        legend_labels={
            "A": "A: low dist., matching title",
            "C": "B: low dist., non-matching title",
            "B": "C: high dist., matching title",
            "D": "D: high dist., non-matching title",
        },
        version_hatches={"A": "", "C": "xx", "B": "///", "D": "\\\\"},
        legend_fontsize=16,
        data_builder=build_exp3_items,
    ),
}


@dataclass
class WordScore:
    word: str
    surprisal: float


class SurprisalScorer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

        if self.model is not None:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.hf_id,
            revision=self.spec.revision,
        )

        common_kwargs = {"revision": self.spec.revision} if self.spec.revision else {}
        dtype = getattr(torch, self.spec.torch_dtype)
        if self.spec.architecture == "masked":
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.spec.hf_id, torch_dtype=dtype, **common_kwargs
            )
            self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.spec.hf_id,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=dtype,
                **common_kwargs,
            )
            if self.device != "cuda":
                self.model.to(self.device)
        self.model.eval()

    def score_words(self, text: str) -> list[WordScore]:
        import torch

        self.load()
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        input_device = self.model.get_input_embeddings().weight.device
        input_ids = input_ids.to(input_device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = input_ids[:, 1:].to(shift_logits.device).contiguous()

        log_probabilities = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probabilities.gather(2, shift_tokens.unsqueeze(-1)).squeeze(-1)
        token_surprisals = -token_log_probs

        token_ids = shift_tokens[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self._aggregate_tokens(tokens, token_surprisals[0].tolist())

    def _aggregate_tokens(self, tokens: list[str], surprisals: list[float]) -> list[WordScore]:
        starts_new = lambda token: token.startswith("Ġ") or token.startswith("▁")

        words: list[WordScore] = []
        current_word = ""
        current_surprisal = 0.0

        for token, surprisal in zip(tokens, surprisals):
            if starts_new(token):
                if current_word:
                    words.append(WordScore(current_word, current_surprisal))
                current_word = token[1:]
                current_surprisal = float(surprisal)
            else:
                current_word += token
                current_surprisal += float(surprisal)

        if current_word:
            words.append(WordScore(current_word, current_surprisal))
        return words


def _generate_exp1_or_exp2(
    root: Path,
    spec: ExperimentSpec,
    scorer: SurprisalScorer,
) -> list[float]:
    values: list[float] = []
    for item in spec.data_builder(root):
        scored_words = scorer.score_words("a" + str(item["text"]))
        target_index = _last_occurrence_surprisal_index(str(item["text"]), str(item["cue_word"])) + 1
        values.append(float(scored_words[target_index].surprisal))
    return values


def _generate_exp3(root: Path, scorer: SurprisalScorer) -> list[float]:
    values: list[float] = []
    for item in build_exp3_items(root):
        scorer.load()
        scored_words = scorer.score_words(str(item["text"]))
        anaphor_text = str(item["anaphor_text"])
        n_ana_tokens = len(
            scorer.tokenizer.encode(" " + anaphor_text, add_special_tokens=False)
        )
        tail = scored_words[-n_ana_tokens:]
        values.append(float(sum(word.surprisal for word in tail) / len(tail)))
    return values


def generate_experiment_wide(
    root: Path,
    experiment: str,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    spec = EXPERIMENTS[experiment]
    models = model_names or MODEL_ORDER
    series_by_model: dict[str, list[float]] = {}

    for model_name in models:
        scorer = SurprisalScorer(model_name)
        if experiment in {"exp1", "exp2"}:
            series_by_model[model_name] = _generate_exp1_or_exp2(root, spec, scorer)
        else:
            series_by_model[model_name] = _generate_exp3(root, scorer)

    return pd.DataFrame(series_by_model)


def load_published_wide(root: Path, experiment: str) -> pd.DataFrame:
    candidates = [
        root / "data" / f"{experiment}.csv",
        root / "generated" / "data" / f"{experiment}.csv",
        root / "data_1" / f"{experiment}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find reference CSV for {experiment!r}. Looked in: {searched}"
    )


def wide_to_long(spec: ExperimentSpec, wide_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    row_index = 0
    for passage_id in spec.passage_ids:
        for version in spec.legacy_version_order:
            row = wide_df.iloc[row_index]
            for model_name in wide_df.columns:
                records.append(
                    {
                        "passage_id": passage_id,
                        "version": version,
                        "model": model_name,
                        "surprisal": float(row[model_name]),
                    }
                )
            row_index += 1
    return pd.DataFrame(records)


def normalize_for_plot(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["normalized_surprisal"] = (
        df.groupby(["passage_id", "model"])["surprisal"]
        .transform(
            lambda values: (
                (values - values.min()) / (values.max() - values.min())
                if values.max() != values.min()
                else 0.0
            )
        )
        .astype(float)
    )
    return df


def summarize_for_plot(long_df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_for_plot(long_df)
    summary = (
        normalized.groupby(["version", "model"])["normalized_surprisal"]
        .agg(mean="mean", sem=_sem)
        .reset_index()
    )
    return summary


def save_experiment_outputs(
    root: Path,
    experiment: str,
    wide_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{experiment}.csv"
    long_path = output_dir / f"{experiment}_long.csv"

    wide_df.to_csv(csv_path, index=False)
    wide_to_long(EXPERIMENTS[experiment], wide_df).to_csv(long_path, index=False)
    return csv_path, long_path


def validate_against_reference(
    generated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> dict[str, float]:
    if list(generated_df.columns) != list(reference_df.columns):
        return {"shape_match": 0.0, "max_abs_diff": float("nan")}
    if generated_df.shape != reference_df.shape:
        return {"shape_match": 0.0, "max_abs_diff": float("nan")}
    diff = (generated_df - reference_df).abs().to_numpy(dtype=float)
    return {"shape_match": 1.0, "max_abs_diff": float(np.nanmax(diff))}


def plot_surprisal_figure(
    experiment: str,
    wide_df: pd.DataFrame,
    figure_dir: Path,
    include_roberta: bool = False,
) -> tuple[Path, Path]:
    spec = EXPERIMENTS[experiment]
    summary = summarize_for_plot(wide_to_long(spec, wide_df))
    models = MODEL_ORDER if include_roberta else PAPER_MODEL_ORDER
    models = [model for model in models if model in summary["model"].unique()]

    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    for offset, version in enumerate(spec.paper_version_order):
        subset = summary[summary["version"] == version].set_index("model").reindex(models)
        ax.bar(
            x + offset * width,
            subset["mean"].values,
            width=width,
            yerr=subset["sem"].values,
            capsize=4,
            label=version,
            alpha=0.90,
            edgecolor="black",
            linewidth=0.6,
            hatch=spec.version_hatches.get(version, ""),
            error_kw=dict(ecolor="black", lw=0.8, capsize=4),
        )

    ax.set_xticks(x + width * (len(spec.paper_version_order) - 1) / 2)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[model] for model in models])
    ax.set_ylabel("Mean surprisal", fontsize=18)
    ax.set_ylim(0, 1.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [spec.legend_labels[label] for label in labels],
        ncol=2,
        fontsize=spec.legend_fontsize,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    prefix = f"fig{spec.figure_number}_surprisal_acl_style"
    pdf_path = figure_dir / f"{prefix}.pdf"
    png_path = figure_dir / f"{prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path
