from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY_DIRS = {
    1: "study 1 (context)",
    2: "study 2 (semantics)",
    3: "study3",
}

STUDY_MATERIAL_DIRS = {
    1: Path("study 1 (context)") / "1 materials" / "passages",
    2: Path("study 2 (semantics)") / "2 materials" / "passages",
}

ANSWER_FILES = {
    1: Path("study 1 (context)") / "1 materials" / "scoring" / "question-scoring.xlsx",
    2: Path("study 2 (semantics)") / "2 materials" / "scoring" / "quest-scoring.xlsx",
    3: Path("anaphor-automate-comprehesion-scoring") / "study3_answers" / "answers.csv",
}

COMPREHENSION_MODEL_ORDER = [
    "gpt2xl",
    "llama3",
    "pythia",
    "mistral7b",
    "mistral24b",
    "falcon-7b-instruct",
    "falcon-instruct-7b",
]

PAPER_COMPREHENSION_MODEL_ORDER = [
    "gpt2xl",
    "llama3",
    "pythia",
    "mistral7b",
    "mistral24b",
]

COMPREHENSION_MODEL_DISPLAY_NAMES = {
    "gpt2xl": "GPT2-XL",
    "llama3": "LLaMa-3.1-8B",
    "pythia": "Pythia-12B",
    "mistral7b": "Mistral-7B",
    "mistral24b": "Mistral-24B",
    "falcon-7b-instruct": "Falcon-7B-Instruct",
    "falcon-instruct-7b": "Falcon-7B-Instruct",
}

COMPREHENSION_VERSION_ORDERS = {
    1: ["A", "C", "B", "D"],
    2: ["B", "A", "D", "C"],
    3: ["A", "C", "B", "D"],
}

COMPREHENSION_VERSION_HATCHES = {
    1: {"A": "", "C": "///", "B": "XX", "D": "\\\\"},
    2: {"B": "", "A": "///", "D": "xx", "C": "\\\\"},
    3: {"A": "", "C": "xx", "B": "///", "D": "\\\\"},
}

COMPREHENSION_LEGEND_LABELS = {
    1: {
        "A": "A: near spat. dist., short temp. dur.",
        "C": "B: near spat. dist, long temp. dur.",
        "B": "C: far spat. dist., short temp. dur.",
        "D": "D: far spat. dist., long temp. dur.",
    },
    2: {
        "B": "A: typ. ante., atyp. distractor",
        "A": "B: typ. ante., typ. distractor",
        "D": "C: atyp. ante., atyp. distractor",
        "C": "D: atyp. ante., typ. distractor",
    },
    3: {
        "A": "A: low dist., matching title",
        "C": "B: low dist., non-matching title",
        "B": "C: high dist., matching title",
        "D": "D: high dist., non-matching title",
    },
}

COMPREHENSION_LEGEND_FONTSIZES = {1: 15, 2: 16, 3: 16}
COMPREHENSION_LEGEND_ANCHORS = {1: (0.5, 1.0), 2: (0.52, 1.0), 3: (0.52, 1.2)}


@dataclass(frozen=True)
class ComprehensionFigurePaths:
    by_model_version_csv: Path
    model_version_png: Path


def _sem(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) <= 1:
        return np.nan
    return float(numeric.std(ddof=1) / np.sqrt(len(numeric)))


def _read_tabbed_text(file_path: Path) -> str:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    text_lines = []
    for line in lines:
        if "\t" in line:
            text = line.split("\t", 1)[1].strip()
            if text:
                text_lines.append(text)
    return " ".join(" ".join(text_lines).split())


def read_passage_text(file_path: str | Path) -> str:
    return _read_tabbed_text(Path(file_path))


def read_questions(file_path: str | Path) -> list[str]:
    lines = Path(file_path).read_text(encoding="utf-8").splitlines()
    questions = []
    for line in lines:
        if "\t" in line:
            question = line.split("\t", 1)[1].strip()
            if question:
                questions.append(question)
    return questions


def load_correct_answers(root: str | Path, study_num: int) -> dict[tuple[Any, ...], str]:
    root = Path(root)
    if study_num in {1, 2}:
        path = root / ANSWER_FILES[study_num]
        df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
        answers: dict[tuple[Any, ...], str] = {}
        for _, row in df.iterrows():
            passage = str(row["Passage"]).zfill(2)
            version = str(row["Version"]).strip().upper()
            question = row["Question"]
            if isinstance(question, str):
                try:
                    question = int(question)
                except ValueError:
                    pass
            answers[(passage, version, question)] = str(row["Answer"]).strip()
        return answers

    if study_num == 3:
        return load_study3_answers(root)

    raise ValueError(f"Unsupported study_num: {study_num}")


def load_study3_answers(root: str | Path, answers_file: str | Path | None = None) -> dict[tuple[int, int], str]:
    root = Path(root)
    path = Path(answers_file) if answers_file else root / ANSWER_FILES[3]
    content = path.read_text(encoding="utf-8").strip()
    try:
        answers_list = ast.literal_eval("[" + content + "]")
    except (SyntaxError, ValueError):
        answers_list = [item.strip().strip("'\"") for item in content.split(",")]

    answers: dict[tuple[int, int], str] = {}
    idx = 0
    for passage_num in range(1, 17):
        for question_num in range(1, 3):
            if idx < len(answers_list):
                answers[(passage_num, question_num)] = str(answers_list[idx]).strip()
            idx += 1
    return answers


def _parse_questions_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _study_material_paths(root: Path, study_num: int) -> tuple[Path, Path]:
    materials_dir = root / STUDY_MATERIAL_DIRS[study_num]
    return materials_dir / "40 column versions", materials_dir / "questions"


def _passage_file_for_version(passages_dir: Path, passage_num: str, version: str) -> Path | None:
    candidates = [
        passages_dir / f"p{passage_num}{version}.txt",
        passages_dir / f"p{passage_num}{version.lower()}.txt",
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_old_response_format(results: pd.DataFrame, study_num: int) -> pd.DataFrame:
    question_ids = {1: 10, 2: 11, 3: 12} if study_num in {1, 2} else {1: 1, 2: 2, 3: 3, 4: 4}
    rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        questions = _parse_questions_cell(row.get("questions", []))
        max_questions = 2 if study_num == 3 else 3
        for question_num in range(1, max_questions + 1):
            question_col = f"question_{question_num}"
            if question_col not in row:
                continue
            rows.append(
                {
                    "study": row.get("study", f"study{study_num}"),
                    "version": str(row.get("version", "")).strip().upper(),
                    "passage_num": int(row["passage_num"]),
                    "passage_key": str(row["passage_num"]).zfill(2),
                    "passage_file": row.get("passage_file", ""),
                    "passage_text": row.get("passage_text", ""),
                    "question_num": question_num,
                    "question_id": question_ids[question_num],
                    "question": questions[question_num - 1] if len(questions) >= question_num else "",
                    "generated_answer": str(row[question_col]).strip(),
                }
            )
    return pd.DataFrame(rows)


def _load_indexed_response_format(root: Path, results: pd.DataFrame, study_num: int) -> pd.DataFrame:
    if study_num not in {1, 2}:
        return _load_study3_indexed_response_format(root, results)

    answer_col = results.columns[-1]
    passages = [f"{i:02d}" for i in range(1, 21) if i != 4]
    versions = ["A", "B", "C", "D"]
    question_ids = {1: 10, 2: 11, 3: 12}
    passages_dir, questions_dir = _study_material_paths(root, study_num)

    rows: list[dict[str, Any]] = []
    answer_idx = 0
    for passage_num in passages:
        for version in versions:
            passage_file = _passage_file_for_version(passages_dir, passage_num, version)
            question_file = questions_dir / f"q{passage_num}.txt"
            if passage_file is None or not question_file.exists():
                continue
            passage_text = read_passage_text(passage_file)
            questions = read_questions(question_file)
            for question_num, question in enumerate(questions, start=1):
                if answer_idx >= len(results):
                    break
                rows.append(
                    {
                        "study": f"study{study_num}",
                        "version": version,
                        "passage_num": int(passage_num),
                        "passage_key": passage_num,
                        "passage_file": passage_file.name,
                        "passage_text": passage_text,
                        "question_num": question_num,
                        "question_id": question_ids[question_num],
                        "question": question,
                        "generated_answer": str(results.iloc[answer_idx][answer_col]).strip(),
                    }
                )
                answer_idx += 1
    return pd.DataFrame(rows)


def _read_study3_passage_data(root: Path, passage_num: int) -> dict[str, Any]:
    path = root / "study3" / f"{passage_num}.txt"
    if not path.exists():
        return {"questions": [], "passage_text": ""}
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = lines[7:11] if len(lines) >= 11 else []
    passage_text = " ".join(line.strip() for line in lines[11:])
    return {"questions": questions, "passage_text": passage_text}


def _load_study3_indexed_response_format(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    versions = ["A", "B", "C", "D"]
    cache: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        csv_passage = int(row["Passage"])
        answer_index = int(row["Answer_Index"])
        if answer_index > 2:
            continue
        passage_num = ((csv_passage - 1) // 4) + 1
        version = versions[(csv_passage - 1) % 4]
        cache.setdefault(passage_num, _read_study3_passage_data(root, passage_num))
        passage_info = cache[passage_num]
        questions = passage_info["questions"]
        rows.append(
            {
                "study": "study3",
                "version": version,
                "passage_num": passage_num,
                "passage_key": str(passage_num).zfill(2),
                "passage_file": f"{passage_num}.txt",
                "passage_text": passage_info["passage_text"],
                "question_num": answer_index,
                "question_id": answer_index,
                "question": questions[answer_index - 1] if len(questions) >= answer_index else "",
                "generated_answer": str(row["Answer"]).strip(),
            }
        )
    return pd.DataFrame(rows)


def load_model_responses(root: str | Path, study_num: int, model_response_file: str | Path) -> pd.DataFrame:
    root = Path(root)
    path = Path(model_response_file)
    if not path.is_absolute():
        path = root / "anaphor-automate-comprehesion-scoring" / "results" / "model_responses" / f"study {study_num}" / path
    results = pd.read_csv(path)

    old_format = {"study", "version", "passage_num"}.issubset(results.columns)
    if old_format:
        return _load_old_response_format(results, study_num)
    return _load_indexed_response_format(root, results, study_num)


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


class GeminiComprehensionScorer:
    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Install google-generativeai to run automated comprehension scoring."
            ) from exc

        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model_name = model_name or os.environ.get("COMPREHENSION_JUDGE_MODEL", "gemini-2.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def score_answer(
        self,
        question: str,
        correct_answer: str,
        generated_answer: str,
        passage_text: str = "",
    ) -> dict[str, Any]:
        prompt = f"""
You are an expert evaluator for reading comprehension questions.
Decide whether the given answer is correct by comparing it to the correct answer.
Use the passage only as context. Return valid JSON only.

QUESTION: {question}
CORRECT ANSWER: {correct_answer}
GIVEN ANSWER: {generated_answer}
PASSAGE CONTEXT: {passage_text[:800]}

Return:
{{"score": 0 or 1, "reasoning": "brief explanation", "is_correct": true or false}}
"""
        try:
            response = self.model.generate_content(contents=prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            result = json.loads(_strip_json_fences(response_text))
            return {
                "score": int(result.get("score", 0)),
                "reasoning": str(result.get("reasoning", "")),
                "is_correct": bool(result.get("is_correct", bool(result.get("score", 0)))),
            }
        except Exception as exc:
            return {"score": 0, "reasoning": f"Error in scoring: {exc}", "is_correct": False}

    def score_passage_batch(
        self,
        passage_data: pd.DataFrame,
        correct_answers: dict[tuple[Any, ...], str],
        study_num: int,
    ) -> pd.DataFrame:
        if passage_data.empty:
            return passage_data.copy()

        passage_key = str(passage_data.iloc[0]["passage_key"])
        passage_num = int(passage_data.iloc[0]["passage_num"])
        version = str(passage_data.iloc[0]["version"]).strip().upper()
        passage_text = str(passage_data.iloc[0].get("passage_text", ""))

        questions_text = []
        for _, row in passage_data.iterrows():
            question_id = row["question_id"]
            correct_key = (passage_num, question_id) if study_num == 3 else (passage_key, version, question_id)
            questions_text.append(
                f"Question {question_id}: {row['question']}\n"
                f"Correct Answer: {correct_answers.get(correct_key, 'Unknown')}\n"
                f"Generated Answer: {row['generated_answer']}"
            )

        prompt = f"""
You are an expert evaluator for reading comprehension questions. Compare each generated answer
to the given correct answer. Use the passage as context, but do not invent new correct answers.

PASSAGE CONTEXT: {passage_text[:1000]}

{chr(10).join(questions_text)}

Return valid JSON only, as an array:
[{{"question_id": 10, "score": 1, "reasoning": "brief explanation", "is_correct": true}}]
"""
        scored = passage_data.copy()
        try:
            response = self.model.generate_content(contents=prompt)
            response_text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(_strip_json_fences(response_text))
            for result in parsed:
                mask = scored["question_id"] == result["question_id"]
                scored.loc[mask, "judge_score"] = int(result.get("score", 0))
                scored.loc[mask, "judge_reasoning"] = str(result.get("reasoning", ""))
                scored.loc[mask, "is_correct"] = bool(result.get("is_correct", bool(result.get("score", 0))))
                correct_key = (
                    (passage_num, result["question_id"])
                    if study_num == 3
                    else (passage_key, version, result["question_id"])
                )
                scored.loc[mask, "correct_answer"] = correct_answers.get(correct_key, "")
        except Exception as exc:
            scored["judge_score"] = 0
            scored["judge_reasoning"] = f"Error: {exc}"
            scored["is_correct"] = False
            scored["correct_answer"] = ""
        return scored


def score_model_responses(
    responses_df: pd.DataFrame,
    correct_answers: dict[tuple[Any, ...], str],
    study_num: int,
    scorer: GeminiComprehensionScorer,
    sleep_seconds: float = 0.5,
) -> pd.DataFrame:
    scored_groups = []
    for _, group in responses_df.groupby(["passage_key", "version"], sort=True):
        scored_groups.append(scorer.score_passage_batch(group, correct_answers, study_num))
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return pd.concat(scored_groups, ignore_index=True) if scored_groups else pd.DataFrame()


def load_scored_comprehension(
    root: str | Path,
    study_num: int,
    scores_dir: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    base = Path(scores_dir) if scores_dir else root / "anaphor-automate-comprehesion-scoring" / "results" / "automate_scores" / f"study {study_num}"
    frames = []
    for path in sorted(base.glob("*.csv")):
        if path.name.startswith("."):
            continue
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df["model"] = path.stem
        df["study_num"] = study_num
        if "judge_score" not in df.columns and "gpt4_score" in df.columns:
            df["judge_score"] = df["gpt4_score"]
        if "judge_reasoning" not in df.columns and "gpt4_reasoning" in df.columns:
            df["judge_reasoning"] = df["gpt4_reasoning"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    scored = pd.concat(frames, ignore_index=True)
    scored["version"] = scored["version"].astype(str).str.strip().str.upper()
    scored["passage_num"] = pd.to_numeric(scored["passage_num"], errors="coerce").astype("Int64")
    scored["judge_score"] = pd.to_numeric(scored["judge_score"], errors="coerce")
    return scored


def calculate_passage_accuracy(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["study_num", "model", "study", "version", "passage_num", "passage_file"] if col in scored_df.columns]
    summary = (
        scored_df.groupby(group_cols, dropna=False)["judge_score"]
        .agg(avg_score="mean", total_score="sum", num_questions="count")
        .reset_index()
    )
    summary["accuracy"] = summary["avg_score"]
    return summary


def summarize_accuracy_by_version(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
    return (
        scored_df.groupby(["study_num", "model", "version"])["judge_score"]
        .agg(mean="mean", sem=_sem, n="count")
        .reset_index()
    )


def summarize_accuracy_by_passage(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame()
    return (
        scored_df.groupby(["study_num", "model", "passage_num", "version"])["judge_score"]
        .agg(mean="mean", sem=_sem, n="count")
        .reset_index()
    )


def _study_num_from_frame(df: pd.DataFrame) -> int | None:
    if "study_num" not in df.columns or df["study_num"].dropna().empty:
        return None
    return int(df["study_num"].dropna().iloc[0])


def _ordered_models(
    models: pd.Series,
    model_order: list[str] | None = None,
    include_extra_models: bool = False,
) -> list[str]:
    unique = list(dict.fromkeys(str(model) for model in models.dropna()))
    order = model_order or PAPER_COMPREHENSION_MODEL_ORDER
    ordered = [model for model in order if model in unique]
    if include_extra_models:
        ordered += [model for model in unique if model not in ordered]
    return ordered


def _ordered_versions(df: pd.DataFrame, study_num: int | None = None) -> list[str]:
    unique = set(str(version) for version in df["version"].dropna().unique())
    order = COMPREHENSION_VERSION_ORDERS.get(study_num or -1, ["A", "B", "C", "D"])
    return [version for version in order if version in unique]


def _apply_acl_axis_style(ax: plt.Axes) -> None:
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def _add_comprehension_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    study_num: int | None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    legend_labels = COMPREHENSION_LEGEND_LABELS.get(study_num or -1, {})
    ax.legend(
        handles,
        [legend_labels.get(label, label) for label in labels],
        ncol=2,
        fontsize=COMPREHENSION_LEGEND_FONTSIZES.get(study_num or -1, 15),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=COMPREHENSION_LEGEND_ANCHORS.get(study_num or -1, (0.5, 1.0)),
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_accuracy_by_model_version(
    summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    model_order: list[str] | None = None,
    include_extra_models: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    study_num = _study_num_from_frame(summary_df)
    models = _ordered_models(summary_df["model"], model_order, include_extra_models)
    versions = _ordered_versions(summary_df, study_num)
    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    _apply_acl_axis_style(ax)
    for offset, version in enumerate(versions):
        subset = summary_df[summary_df["version"] == version].set_index("model").reindex(models)
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
            hatch=COMPREHENSION_VERSION_HATCHES.get(study_num or -1, {}).get(version, ""),
            error_kw=dict(ecolor="black", lw=0.8, capsize=4),
        )
    ax.set_xticks(x + width * (len(versions) - 1) / 2)
    ax.set_xticklabels([COMPREHENSION_MODEL_DISPLAY_NAMES.get(model, model) for model in models])
    ax.set_ylabel("Mean accuracy", fontsize=18)
    ax.set_ylim(0, 1.05)

    _add_comprehension_legend(fig, ax, study_num)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_accuracy_by_passage_version(
    scored_or_summary_df: pd.DataFrame,
    output_path: str | Path | None = None,
    model: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    df = scored_or_summary_df.copy()
    study_num = _study_num_from_frame(df)
    if model is not None and "model" in df.columns:
        df = df[df["model"] == model]
    if "mean" not in df.columns:
        df = summarize_accuracy_by_passage(df)
    if model is None and "model" in df.columns:
        df = df[df["model"].isin(PAPER_COMPREHENSION_MODEL_ORDER)]
    grouped = df.groupby(["passage_num", "version"])["mean"].mean().reset_index()
    versions = _ordered_versions(grouped, study_num)
    pivot = grouped.pivot(index="passage_num", columns="version", values="mean").sort_index()
    pivot = pivot.reindex(columns=versions)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    _apply_acl_axis_style(ax)
    pivot.plot(
        kind="bar",
        ax=ax,
        width=0.82,
        edgecolor="black",
        linewidth=0.6,
    )
    for container, version in zip(ax.containers, versions):
        for patch in container.patches:
            patch.set_hatch(COMPREHENSION_VERSION_HATCHES.get(study_num or -1, {}).get(version, ""))
            patch.set_alpha(0.90)
    ax.set_ylabel("Mean accuracy", fontsize=18)
    ax.set_xlabel("Passage", fontsize=18)
    ax.set_ylim(0, 1.05)
    _add_comprehension_legend(fig, ax, study_num)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_question_accuracy_by_version(
    scored_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    scored_df = scored_df[scored_df["model"].isin(PAPER_COMPREHENSION_MODEL_ORDER)].copy()
    study_num = _study_num_from_frame(scored_df)
    grouped = (
        scored_df.groupby(["question_id", "version"])["judge_score"]
        .agg(mean="mean", sem=_sem)
        .reset_index()
    )
    versions = _ordered_versions(grouped, study_num)
    questions = sorted(grouped["question_id"].dropna().unique())
    x = np.arange(len(questions))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    _apply_acl_axis_style(ax)
    for offset, version in enumerate(versions):
        subset = grouped[grouped["version"] == version].set_index("question_id").reindex(questions)
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
            hatch=COMPREHENSION_VERSION_HATCHES.get(study_num or -1, {}).get(version, ""),
            error_kw=dict(ecolor="black", lw=0.8, capsize=4),
        )
    ax.set_ylabel("Mean accuracy", fontsize=18)
    ax.set_xlabel("Question", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width * (len(versions) - 1) / 2)
    ax.set_xticklabels([str(question) for question in questions])
    _add_comprehension_legend(fig, ax, study_num)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def save_comprehension_outputs(
    scored_df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> ComprehensionFigurePaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_scored_df = scored_df[scored_df["model"].isin(PAPER_COMPREHENSION_MODEL_ORDER)].copy()
    by_model = summarize_accuracy_by_version(paper_scored_df)
    by_model_csv = output_dir / f"{prefix}_accuracy_by_model_version.csv"
    model_png = output_dir / f"{prefix}_accuracy_by_model_version.png"
    by_model.to_csv(by_model_csv, index=False)
    plot_accuracy_by_model_version(by_model, model_png)
    plt.close("all")
    return ComprehensionFigurePaths(by_model_csv, model_png)
