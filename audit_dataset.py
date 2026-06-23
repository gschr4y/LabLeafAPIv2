"""
Auditoria rapida de vazamento/memorizacao para datasets ImageFolder.

Exemplos:
  py audit_dataset.py --data-dir data
  py audit_dataset.py --data-dir data_clean --json-output data_clean/audit_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifica sinais simples de vazamento entre treino/validacao/teste."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data_clean"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--max-hash-images",
        type=int,
        default=10000,
        help="Limite de imagens para checagem de hash exato.",
    )
    return parser.parse_args()


def clean_stem(path: Path) -> str:
    stem = path.stem.lower()
    while stem.startswith("aug_"):
        stem = stem[4:]
    return stem


def iter_image_paths(data_dir: Path) -> Iterable[Tuple[str, Path]]:
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limited_paths(paths: List[Path], limit: int = 10) -> List[str]:
    return [str(path) for path in paths[:limit]]


def audit_dataset(data_dir: Path, max_hash_images: int) -> Dict[str, object]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {data_dir.resolve()}")

    records: List[Dict[str, object]] = []
    counts: Dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for split, path in iter_image_paths(data_dir):
        class_name = path.parent.name
        counts[split][class_name] += 1
        records.append(
            {
                "split": split,
                "class_name": class_name,
                "path": path,
                "file_name": path.name,
                "clean_stem": clean_stem(path),
                "is_augmented": path.stem.lower().startswith("aug_"),
            }
        )

    issues: List[str] = []
    warnings: List[str] = []

    if data_dir.name.lower() in {"data", "data_augmented", "data_augmented_max"}:
        issues.append(
            f"pasta '{data_dir.name}' e uma origem insegura neste projeto; prefira data_clean."
        )

    existing_splits = {record["split"] for record in records}
    for required in ("train", "val", "test"):
        if required not in existing_splits:
            issues.append(f"split {required}/ ausente.")

    augmented = [record["path"] for record in records if record["is_augmented"]]
    if augmented:
        issues.append(f"{len(augmented)} arquivos aug_* encontrados.")

    by_class_stem: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    by_stem: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_class_stem[(str(record["class_name"]), str(record["clean_stem"]))].append(record)
        by_stem[str(record["clean_stem"])].append(record)

    cross_split = []
    for (class_name, stem), group in by_class_stem.items():
        splits = sorted({str(record["split"]) for record in group})
        if len(splits) > 1:
            cross_split.append(
                {
                    "class_name": class_name,
                    "clean_stem": stem,
                    "splits": splits,
                    "paths": limited_paths([record["path"] for record in group]),
                }
            )
    if cross_split:
        issues.append(f"{len(cross_split)} nomes-base aparecem em mais de um split.")

    cross_class = []
    for stem, group in by_stem.items():
        classes = sorted({str(record["class_name"]) for record in group})
        if len(classes) > 1:
            cross_class.append(
                {
                    "clean_stem": stem,
                    "classes": classes,
                    "paths": limited_paths([record["path"] for record in group]),
                }
            )
    if cross_class:
        issues.append(f"{len(cross_class)} nomes-base aparecem em classes diferentes.")

    duplicate_hash_groups: List[Dict[str, object]] = []
    if len(records) <= max_hash_images:
        by_hash: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for record in records:
            path = record["path"]
            by_hash[file_sha256(path)].append(record)
        for digest, group in by_hash.items():
            splits = sorted({str(record["split"]) for record in group})
            if len(group) > 1 and len(splits) > 1:
                duplicate_hash_groups.append(
                    {
                        "sha256": digest,
                        "splits": splits,
                        "paths": limited_paths([record["path"] for record in group]),
                    }
                )
        if duplicate_hash_groups:
            issues.append(
                f"{len(duplicate_hash_groups)} grupos de hash exato cruzam splits."
            )
    else:
        warnings.append(
            f"hash exato pulado: {len(records)} imagens excedem max_hash_images={max_hash_images}."
        )

    small_train_classes = {
        class_name: count
        for class_name, count in counts["train"].items()
        if count < 10
    }
    if small_train_classes:
        warnings.append(
            "classes com menos de 10 imagens em train: "
            + ", ".join(f"{name}={count}" for name, count in sorted(small_train_classes.items()))
        )

    split_summary = {
        split: {
            "total": sum(counter.values()),
            "count_by_class": dict(sorted(counter.items())),
        }
        for split, counter in counts.items()
    }
    return {
        "data_dir": str(data_dir.resolve()),
        "total_images": len(records),
        "safe_for_training": not issues,
        "issues": issues,
        "warnings": warnings,
        "split_summary": split_summary,
        "examples": {
            "augmented": limited_paths(augmented),
            "cross_split": cross_split[:10],
            "cross_class": cross_class[:10],
            "duplicate_hash_groups": duplicate_hash_groups[:10],
        },
    }


def print_report(report: Dict[str, object]) -> None:
    print("\nAuditoria do dataset")
    print("-" * 72)
    print(f"Dataset : {report['data_dir']}")
    print(f"Imagens : {report['total_images']}")
    print(f"Status  : {'LIMPO' if report['safe_for_training'] else 'RISCO'}")
    print("-" * 72)

    for split, split_info in report["split_summary"].items():  # type: ignore[union-attr]
        print(f"{split:>5}: {split_info['total']:4d}")  # type: ignore[index]

    issues = report["issues"]  # type: ignore[assignment]
    warnings = report["warnings"]  # type: ignore[assignment]
    if issues:
        print("\nProblemas que bloqueiam treino confiavel:")
        for issue in issues:
            print(f"- {issue}")
    if warnings:
        print("\nAvisos:")
        for warning in warnings:
            print(f"- {warning}")
    print("-" * 72)


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.data_dir, args.max_hash_images)
    print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2, default=str)
        print(f"JSON: {args.json_output.resolve()}")
    raise SystemExit(0 if report["safe_for_training"] else 2)


if __name__ == "__main__":
    main()
