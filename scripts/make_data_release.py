from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DATA_PATHS = [
    "LICENSE-DATA.md",
    "README.md",
    "data",
    "data_1",
    "filled_words.csv",
    "generated",
    "study 1 (context)",
    "study 2 (semantics)",
    "study1",
    "study2",
    "study3",
    "anaphor-automate-comprehesion-scoring/results",
    "anaphor-automate-comprehesion-scoring/study3_answers",
]

IGNORED_PARTS = {".DS_Store", "__pycache__", ".ipynb_checkpoints"}


def should_include(path: Path) -> bool:
    return not any(part in IGNORED_PARTS for part in path.parts)


def iter_files(root: Path, relative_path: str) -> list[Path]:
    path = root / relative_path
    if not path.exists():
        return []
    if path.is_file():
        return [path] if should_include(path) else []
    return sorted(item for item in path.rglob("*") if item.is_file() and should_include(item))


def build_manifest(root: Path, files: list[Path]) -> str:
    lines = [
        "# Data Package Manifest",
        "",
        "License: Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "License URL: https://creativecommons.org/licenses/by/4.0/",
        "",
        "Files:",
    ]
    lines.extend(f"- {path.relative_to(root)}" for path in files)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a CC BY 4.0 data release zip for the CogSci 2026 reproduction package."
    )
    parser.add_argument(
        "--output",
        default="dist/anaphor-cogsci-2026-data.zip",
        help="Path for the generated zip archive.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for relative_path in DATA_PATHS:
        files.extend(iter_files(root, relative_path))

    manifest = build_manifest(root, files)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
        archive.writestr("DATA_MANIFEST.md", manifest)

    print(output_path)


if __name__ == "__main__":
    main()
