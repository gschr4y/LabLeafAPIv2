"""
Prepara um dataset limpo para classificacao de folhas de soja.

O script usa como origem o dataset atual em data/, mas:
- ignora arquivos gerados por aumento offline, como aug_*.jpg;
- remove grupos com o mesmo nome-base aparecendo em classes diferentes;
- cria train/val/test novos, com copia das imagens reais;
- salva relatorios JSON para auditoria.

Exemplo:
  py prepare_clean_dataset.py --source data --output data_clean --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class ImageItem:
    source_path: str
    original_split: str
    class_name: str
    file_name: str
    clean_stem: str
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstrui um split limpo train/val/test a partir do dataset atual."
    )
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data_clean"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apaga a pasta de saida antes de recriar o dataset limpo.",
    )
    parser.add_argument(
        "--keep-conflicting-stems",
        action="store_true",
        help=(
            "Mantem imagens cujo mesmo nome-base aparece em classes diferentes. "
            "Por padrao elas sao removidas por seguranca."
        ),
    )
    return parser.parse_args()


def iter_image_paths(source: Path) -> Iterable[Tuple[str, Path]]:
    for split in ("train", "val", "test"):
        split_dir = source / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield split, path


def is_offline_augmented(path: Path) -> bool:
    return path.stem.lower().startswith("aug_")


def clean_stem(path: Path) -> str:
    stem = path.stem.lower()
    while stem.startswith("aug_"):
        stem = stem[4:]
    return stem


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_items(source: Path) -> Tuple[List[ImageItem], List[Dict[str, str]]]:
    items: List[ImageItem] = []
    skipped_augmented: List[Dict[str, str]] = []

    for split, path in iter_image_paths(source):
        class_name = path.parent.name
        record = {
            "path": str(path),
            "original_split": split,
            "class_name": class_name,
            "file_name": path.name,
            "clean_stem": clean_stem(path),
        }
        if is_offline_augmented(path):
            skipped_augmented.append(record)
            continue

        items.append(
            ImageItem(
                source_path=str(path),
                original_split=split,
                class_name=class_name,
                file_name=path.name,
                clean_stem=clean_stem(path),
                sha256=file_sha256(path),
            )
        )

    return items, skipped_augmented


def remove_exact_duplicates(items: List[ImageItem]) -> Tuple[List[ImageItem], List[List[Dict[str, str]]]]:
    by_hash: Dict[str, List[ImageItem]] = defaultdict(list)
    for item in items:
        by_hash[item.sha256].append(item)

    kept: List[ImageItem] = []
    duplicate_groups: List[List[Dict[str, str]]] = []
    for group in by_hash.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        ordered = sorted(group, key=lambda item: item.source_path)
        kept.append(ordered[0])
        duplicate_groups.append([asdict(item) for item in ordered])

    return kept, duplicate_groups


def remove_conflicting_stems(
    items: List[ImageItem],
) -> Tuple[List[ImageItem], List[Dict[str, object]]]:
    by_stem: Dict[str, List[ImageItem]] = defaultdict(list)
    for item in items:
        by_stem[item.clean_stem].append(item)

    conflicting_stems = {
        stem: group
        for stem, group in by_stem.items()
        if len({item.class_name for item in group}) > 1
    }
    conflicts = [
        {
            "clean_stem": stem,
            "classes": sorted({item.class_name for item in group}),
            "items": [asdict(item) for item in sorted(group, key=lambda item: item.source_path)],
        }
        for stem, group in sorted(conflicting_stems.items())
    ]
    clean_items = [
        item
        for item in items
        if item.clean_stem not in conflicting_stems
    ]
    return clean_items, conflicts


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0

    train_count = max(1, round(total * train_ratio))
    val_count = max(1, round(total * val_ratio))
    test_count = total - train_count - val_count

    if test_count < 1:
        test_count = 1
        train_count = max(1, train_count - 1)

    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < total:
        train_count += 1

    return train_count, val_count, test_count


def split_by_class(
    items: List[ImageItem],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[ImageItem]]:
    rng = random.Random(seed)
    by_class_and_stem: Dict[str, Dict[str, List[ImageItem]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        by_class_and_stem[item.class_name][item.clean_stem].append(item)

    result = {"train": [], "val": [], "test": []}
    for class_name, groups_by_stem in sorted(by_class_and_stem.items()):
        groups = list(groups_by_stem.values())
        rng.shuffle(groups)
        total_images = sum(len(group) for group in groups)
        target_counts = split_counts(total_images, train_ratio, val_ratio, test_ratio)
        split_names = ("train", "val", "test")
        class_split_counts = {split: 0 for split in split_names}

        for group in sorted(groups, key=lambda value: (-len(value), value[0].clean_stem)):
            best_split = min(
                split_names,
                key=lambda split: (
                    class_split_counts[split] / max(1, target_counts[split_names.index(split)]),
                    class_split_counts[split],
                    split_names.index(split),
                ),
            )
            result[best_split].extend(group)
            class_split_counts[best_split] += len(group)

    for split_items in result.values():
        split_items.sort(key=lambda item: (item.class_name.lower(), item.file_name.lower()))

    return result


def unique_output_name(item: ImageItem, used_names: set[str]) -> str:
    source = Path(item.source_path)
    base_name = source.name
    candidate = base_name
    if candidate.lower() not in used_names:
        used_names.add(candidate.lower())
        return candidate

    short_hash = item.sha256[:10]
    candidate = f"{source.stem}_{short_hash}{source.suffix.lower()}"
    used_names.add(candidate.lower())
    return candidate


def write_dataset(output: Path, splits: Dict[str, List[ImageItem]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    used_by_folder: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for split, items in splits.items():
        summary[split] = {}
        for item in items:
            class_dir = output / split / item.class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            output_name = unique_output_name(
                item,
                used_by_folder[(split, item.class_name)],
            )
            shutil.copy2(item.source_path, class_dir / output_name)
            summary[split][item.class_name] = summary[split].get(item.class_name, 0) + 1

    return summary


def count_by_class(items: Iterable[ImageItem]) -> Dict[str, int]:
    return dict(sorted(Counter(item.class_name for item in items).items()))


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("As proporcoes precisam somar um valor positivo.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio precisa somar 1.0.")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset de origem nao encontrado: {source}")

    if output.exists():
        if not args.force:
            raise FileExistsError(f"A pasta {output} ja existe. Use --force para recriar.")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    collected, skipped_augmented = collect_items(source)
    deduped, exact_duplicate_groups = remove_exact_duplicates(collected)

    if args.keep_conflicting_stems:
        clean_items = deduped
        conflicting_stems: List[Dict[str, object]] = []
    else:
        clean_items, conflicting_stems = remove_conflicting_stems(deduped)

    splits = split_by_class(
        clean_items,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    split_summary = write_dataset(output, splits)

    report = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "input_images_after_extension_filter": len(collected) + len(skipped_augmented),
        "skipped_offline_augmented": len(skipped_augmented),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "conflicting_stem_groups_removed": len(conflicting_stems),
        "images_available_after_cleaning": len(clean_items),
        "clean_count_by_class": count_by_class(clean_items),
        "split_summary": split_summary,
        "skipped_offline_augmented_examples": skipped_augmented[:50],
        "exact_duplicate_groups_detail": exact_duplicate_groups,
        "conflicting_stem_groups_detail": conflicting_stems,
    }
    with (output / "clean_dataset_report.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print("\nDataset limpo criado")
    print("-" * 72)
    print(f"Origem : {source}")
    print(f"Saida  : {output}")
    print(f"Imagens ignoradas por aug_: {len(skipped_augmented)}")
    print(f"Grupos de hash duplicado  : {len(exact_duplicate_groups)}")
    print(f"Stems conflitantes remov. : {len(conflicting_stems)}")
    print(f"Imagens limpas usadas     : {len(clean_items)}")
    print("-" * 72)
    for split in ("train", "val", "test"):
        total = sum(split_summary.get(split, {}).values())
        print(f"{split:>5}: {total:4d}")
        for class_name, count in sorted(split_summary.get(split, {}).items()):
            print(f"       {class_name:<28s} {count:4d}")
    print("-" * 72)
    print(f"Relatorio: {output / 'clean_dataset_report.json'}")


if __name__ == "__main__":
    main()
