"""
Treinamento completo de ResNet-18 para classificacao de doencas em folhas de soja.

Estrutura esperada do dataset:

data/
  train/
    classe_1/
    classe_2/
  val/
    classe_1/
    classe_2/
  test/              opcional
    classe_1/
    classe_2/

Exemplos:
  py resnet18_soja.py
  py resnet18_soja.py --epochs 50 --batch-size 16
  py resnet18_soja.py --device cpu --dry-run

Dependencias:
  py -m pip install torch torchvision matplotlib seaborn tqdm pillow
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    checkpoint_name: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    num_workers: int
    device: str
    pretrained: bool
    freeze_backbone: bool
    dropout: float
    label_smoothing: float
    use_weighted_sampler: bool
    amp: bool
    dry_run: bool
    prepare_augmented_data: bool
    augmented_data_dir: Path
    augment_target_per_class: int
    force_augment: bool
    augment_only: bool
    stop_at: Optional[str]
    max_train_minutes: Optional[float]
    initial_checkpoint: Optional[Path]


def parse_args() -> TrainConfig:
    default_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(
        description="Treina uma ResNet-18 no dataset local de folhas de soja."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("resultados"))
    parser.add_argument("--checkpoint-name", default="resnet18_soja_best.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda ou dml/directml se torch_directml estiver instalado.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--prepare-augmented-data",
        action="store_true",
        help="Cria uma copia aumentada do dataset antes de treinar.",
    )
    parser.add_argument("--augmented-data-dir", type=Path, default=Path("data_augmented"))
    parser.add_argument(
        "--augment-target-per-class",
        type=int,
        default=180,
        help="Quantidade final desejada por classe em train no dataset aumentado.",
    )
    parser.add_argument(
        "--force-augment",
        action="store_true",
        help="Apaga e recria a pasta do dataset aumentado.",
    )
    parser.add_argument(
        "--augment-only",
        action="store_true",
        help="Apenas cria o dataset aumentado e encerra sem treinar.",
    )
    parser.add_argument(
        "--stop-at",
        default=None,
        help="Horario local para parar com seguranca, exemplo: 17:30.",
    )
    parser.add_argument(
        "--max-train-minutes",
        type=float,
        default=None,
        help="Tempo maximo de treino em minutos.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint usado como ponto de partida, carregando apenas os pesos do modelo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Carrega dados, monta o modelo e executa apenas um forward pass.",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=not args.no_weighted_sampler,
        amp=not args.no_amp,
        dry_run=args.dry_run,
        prepare_augmented_data=args.prepare_augmented_data,
        augmented_data_dir=args.augmented_data_dir,
        augment_target_per_class=args.augment_target_per_class,
        force_augment=args.force_augment,
        augment_only=args.augment_only,
        stop_at=args.stop_at,
        max_train_minutes=args.max_train_minutes,
        initial_checkpoint=args.initial_checkpoint,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def resolve_device(device_name: str) -> Tuple[torch.device, str]:
    name = device_name.lower().strip()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "directml"
        except Exception:
            return torch.device("cpu"), "cpu"

    if name in {"dml", "directml"}:
        import torch_directml  # type: ignore

        return torch_directml.device(), "directml"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi escolhido, mas torch.cuda.is_available() retornou False.")

    return torch.device(name), name


def resolve_training_deadline(config: TrainConfig) -> Tuple[Optional[float], Optional[str]]:
    deadlines: List[datetime] = []
    now = datetime.now()

    if config.stop_at:
        raw = config.stop_at.strip()
        parsed: Optional[datetime] = None
        for fmt in ("%H:%M", "%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                value = datetime.strptime(raw, fmt)
                if fmt.startswith("%H"):
                    parsed = now.replace(
                        hour=value.hour,
                        minute=value.minute,
                        second=value.second,
                        microsecond=0,
                    )
                    if parsed <= now:
                        parsed += timedelta(days=1)
                else:
                    parsed = value
                break
            except ValueError:
                continue

        if parsed is None:
            raise RuntimeError(
                "--stop-at invalido. Use HH:MM, HH:MM:SS ou YYYY-MM-DD HH:MM."
            )
        deadlines.append(parsed)

    if config.max_train_minutes is not None:
        if config.max_train_minutes <= 0:
            raise RuntimeError("--max-train-minutes precisa ser maior que zero.")
        deadlines.append(now + timedelta(minutes=config.max_train_minutes))

    if not deadlines:
        return None, None

    deadline = min(deadlines)
    return deadline.timestamp(), deadline.strftime("%Y-%m-%d %H:%M:%S")


def deadline_reached(deadline_ts: Optional[float]) -> bool:
    return deadline_ts is not None and time.time() >= deadline_ts


def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return {"train": train_tfms, "val": eval_tfms, "test": eval_tfms}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def iter_image_files(folder: Path) -> List[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_offline_augmentation(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(degrees=25, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=8,
                fill=0,
            ),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.04,
            ),
        ]
    )


def prepare_augmented_dataset(config: TrainConfig) -> Path:
    source_dir = config.data_dir.resolve()
    target_dir = config.augmented_data_dir.resolve()

    if source_dir == target_dir:
        raise RuntimeError("A pasta aumentada nao pode ser a mesma pasta do dataset original.")
    if config.augment_target_per_class <= 0:
        raise RuntimeError("--augment-target-per-class precisa ser maior que zero.")
    if not (source_dir / "train").exists():
        raise RuntimeError(f"Nao encontrei a pasta de treino: {source_dir / 'train'}")

    if target_dir.exists():
        if config.force_augment:
            print(f"Removendo dataset aumentado anterior: {target_dir}")
            shutil.rmtree(target_dir)
        elif (target_dir / "train").exists() and (target_dir / "val").exists():
            print(f"Usando dataset aumentado ja existente: {target_dir}")
            return target_dir
        else:
            raise RuntimeError(
                f"A pasta {target_dir} ja existe, mas nao parece ser um dataset completo. "
                "Use --force-augment para recriar."
            )

    print("\nPreparando dataset aumentado")
    print("-" * 72)
    print(f"Origem : {source_dir}")
    print(f"Destino: {target_dir}")
    print(f"Meta   : {config.augment_target_per_class} imagens por classe em train")

    target_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        split_source = source_dir / split
        if split_source.exists():
            shutil.copytree(split_source, target_dir / split)

    train_source = source_dir / "train"
    train_target = target_dir / "train"
    train_target.mkdir(parents=True, exist_ok=True)

    augment = build_offline_augmentation(config.image_size)
    summary = {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "target_per_class": config.augment_target_per_class,
        "classes": {},
    }

    class_dirs = sorted(path for path in train_source.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_target = train_target / class_name
        class_target.mkdir(parents=True, exist_ok=True)

        originals = iter_image_files(class_dir)
        if not originals:
            print(f"  {class_name:<28s} sem imagens, pulando.")
            continue

        for image_path in originals:
            shutil.copy2(image_path, class_target / image_path.name)

        needed = max(0, config.augment_target_per_class - len(originals))
        generated = 0
        attempts = 0

        while generated < needed and attempts < needed * 4:
            attempts += 1
            source_image = originals[generated % len(originals)]
            try:
                with Image.open(source_image) as image:
                    augmented = augment(image.convert("RGB"))
                    out_name = f"aug_{generated + 1:04d}_{source_image.stem}.jpg"
                    augmented.save(class_target / out_name, format="JPEG", quality=92)
                    generated += 1
            except Exception as exc:
                print(f"  Aviso: falha ao aumentar {source_image.name}: {exc}")

        final_count = len(iter_image_files(class_target))
        summary["classes"][class_name] = {
            "original": len(originals),
            "generated": generated,
            "final": final_count,
        }
        print(
            f"  {class_name:<28s} original={len(originals):4d} "
            f"geradas={generated:4d} final={final_count:4d}"
        )

    with (target_dir / "augmentation_resumo.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("-" * 72)
    print(f"Dataset aumentado pronto em: {target_dir}")
    return target_dir


def load_datasets(data_dir: Path, image_size: int) -> Dict[str, datasets.ImageFolder]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Pasta do dataset nao encontrada: {data_dir.resolve()}")

    tfms = build_transforms(image_size)
    loaded: Dict[str, datasets.ImageFolder] = {}

    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if split_dir.exists():
            dataset = datasets.ImageFolder(split_dir, transform=tfms[split])
            if len(dataset) == 0:
                raise RuntimeError(f"A pasta {split_dir} existe, mas nao possui imagens validas.")
            loaded[split] = dataset

    if "train" not in loaded:
        raise RuntimeError(
            "Nao encontrei data/train. Organize o dataset com subpastas por classe "
            "dentro de data/train e data/val."
        )
    if "val" not in loaded:
        raise RuntimeError(
            "Nao encontrei data/val. Separe uma parte do dataset para validacao "
            "em data/val antes de treinar."
        )

    train_classes = loaded["train"].classes
    for split, dataset in loaded.items():
        if dataset.classes != train_classes:
            raise RuntimeError(
                f"As classes de {split} nao batem com train.\n"
                f"train: {train_classes}\n{split}: {dataset.classes}"
            )

    return loaded


def count_by_class(dataset: datasets.ImageFolder) -> Dict[str, int]:
    counts = {class_name: 0 for class_name in dataset.classes}
    for _, class_idx in dataset.samples:
        counts[dataset.classes[class_idx]] += 1
    return counts


def print_dataset_summary(datasets_by_split: Dict[str, datasets.ImageFolder]) -> None:
    print("\nResumo do dataset")
    print("-" * 72)
    for split, dataset in datasets_by_split.items():
        print(f"{split:>5}: {len(dataset):4d} imagens | {len(dataset.classes)} classes")
        counts = count_by_class(dataset)
        for class_name, count in counts.items():
            print(f"       {class_name:<28s} {count:4d}")
    print("-" * 72)


def save_dataset_summary(
    datasets_by_split: Dict[str, datasets.ImageFolder], output_dir: Path
) -> None:
    summary = {
        split: {
            "total": len(dataset),
            "classes": dataset.classes,
            "count_by_class": count_by_class(dataset),
        }
        for split, dataset in datasets_by_split.items()
    }
    with (output_dir / "dataset_resumo.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    train_dataset = datasets_by_split["train"]
    class_to_idx = train_dataset.class_to_idx
    with (output_dir / "classes.json").open("w", encoding="utf-8") as fp:
        json.dump(class_to_idx, fp, ensure_ascii=False, indent=2)


def make_weighted_sampler(
    dataset: datasets.ImageFolder, generator: torch.Generator
) -> WeightedRandomSampler:
    labels = torch.tensor([target for _, target in dataset.samples], dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=len(dataset.classes)).float()
    class_weights = 1.0 / class_counts.clamp_min(1.0)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def make_loaders(
    datasets_by_split: Dict[str, datasets.ImageFolder],
    config: TrainConfig,
    device_label: str,
) -> Dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    loaders: Dict[str, DataLoader] = {}

    for split, dataset in datasets_by_split.items():
        sampler = None
        shuffle = split == "train"

        if split == "train" and config.use_weighted_sampler:
            sampler = make_weighted_sampler(dataset, generator)
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=device_label == "cuda",
            worker_init_fn=seed_worker if config.num_workers > 0 else None,
            persistent_workers=config.num_workers > 0,
        )

    return loaders


def build_model(
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except AttributeError:
            weights = "legacy_pretrained"

    try:
        if weights == "legacy_pretrained":
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(weights=weights)
    except Exception as exc:
        print(f"Aviso: nao foi possivel carregar pesos pre-treinados ({exc}).")
        print("Continuando com ResNet-18 inicializada do zero.")
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.75),
        nn.Linear(256, num_classes),
    )

    return model.to(device)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item() / labels.size(0)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    deadline_ts: Optional[float] = None,
) -> Tuple[float, float, bool]:
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total = 0
    interrupted = False

    progress = tqdm(loader, desc="train", ncols=100, leave=False)
    for images, labels in progress:
        if deadline_reached(deadline_ts):
            interrupted = True
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(logits.detach(), labels) * batch_size
        total += batch_size

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{running_correct / total:.4f}",
        )

        if deadline_reached(deadline_ts):
            interrupted = True
            break

    if total == 0:
        return 0.0, 0.0, interrupted
    return running_loss / total, running_correct / total, interrupted


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    total = 0

    progress = tqdm(loader, desc=split_name, ncols=100, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += accuracy_from_logits(logits, labels) * batch_size
        total += batch_size

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{running_correct / total:.4f}",
        )

    return running_loss / total, running_correct / total


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_acc: float,
    class_names: List[str],
    config: TrainConfig,
    history: List[Dict[str, float]],
) -> Dict[str, object]:
    config_dict = asdict(config)
    config_dict["data_dir"] = str(config.data_dir)
    config_dict["output_dir"] = str(config.output_dir)

    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "classes": class_names,
        "config": config_dict,
        "history": history,
    }


def save_history_csv(history: List[Dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    path = output_dir / "historico_treinamento.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: List[Dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="treino")
    axes[0].plot(epochs, val_loss, marker="o", label="validacao")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="treino")
    axes[1].plot(epochs, val_acc, marker="o", label="validacao")
    axes[1].set_title("Acuracia")
    axes[1].set_xlabel("Epoca")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "curvas_treinamento.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> Tuple[List[int], List[int]]:
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []

    for images, labels in tqdm(loader, desc=f"avaliando_{split_name}", ncols=100):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()

        preds_all.extend(preds)
        labels_all.extend(labels.tolist())

    return labels_all, preds_all


def compute_accuracy(labels: List[int], preds: List[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(real == pred) for real, pred in zip(labels, preds))
    return correct / len(labels)


def compute_confusion_matrix(
    labels: List[int],
    preds: List[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for real, pred in zip(labels, preds):
        if 0 <= real < num_classes and 0 <= pred < num_classes:
            matrix[real, pred] += 1
    return matrix


def build_classification_report(
    labels: List[int],
    preds: List[int],
    class_names: List[str],
) -> str:
    cm = compute_confusion_matrix(labels, preds, len(class_names))
    total = cm.sum()
    accuracy = np.trace(cm) / total if total else 0.0

    rows: List[Tuple[str, float, float, float, int]] = []
    for idx, class_name in enumerate(class_names):
        true_positive = float(cm[idx, idx])
        false_positive = float(cm[:, idx].sum() - cm[idx, idx])
        false_negative = float(cm[idx, :].sum() - cm[idx, idx])
        support = int(cm[idx, :].sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        rows.append((class_name, precision, recall, f1, support))

    macro_precision = float(np.mean([row[1] for row in rows])) if rows else 0.0
    macro_recall = float(np.mean([row[2] for row in rows])) if rows else 0.0
    macro_f1 = float(np.mean([row[3] for row in rows])) if rows else 0.0

    supports = np.array([row[4] for row in rows], dtype=float)
    if supports.sum() > 0:
        weighted_precision = float(np.average([row[1] for row in rows], weights=supports))
        weighted_recall = float(np.average([row[2] for row in rows], weights=supports))
        weighted_f1 = float(np.average([row[3] for row in rows], weights=supports))
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    name_width = max(18, max(len(name) for name in class_names) if class_names else 0)
    header = (
        f"{'classe':<{name_width}} {'precision':>10} {'recall':>10} "
        f"{'f1-score':>10} {'support':>10}"
    )
    lines = [header, "-" * len(header)]
    for class_name, precision, recall, f1, support in rows:
        lines.append(
            f"{class_name:<{name_width}} {precision:10.4f} {recall:10.4f} "
            f"{f1:10.4f} {support:10d}"
        )

    lines.append("")
    lines.append(f"{'accuracy':<{name_width}} {'':>10} {'':>10} {accuracy:10.4f} {int(total):10d}")
    lines.append(
        f"{'macro avg':<{name_width}} {macro_precision:10.4f} {macro_recall:10.4f} "
        f"{macro_f1:10.4f} {int(total):10d}"
    )
    lines.append(
        f"{'weighted avg':<{name_width}} {weighted_precision:10.4f} {weighted_recall:10.4f} "
        f"{weighted_f1:10.4f} {int(total):10d}"
    )
    return "\n".join(lines)


def save_final_report(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    output_dir: Path,
    device: torch.device,
    split_name: str,
) -> float:
    labels, preds = collect_predictions(model, loader, device, split_name)

    acc = compute_accuracy(labels, preds)
    report = build_classification_report(labels, preds, class_names)

    report_path = output_dir / f"relatorio_{split_name}.txt"
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write(f"Acuracia {split_name}: {acc:.6f}\n\n")
        fp.write(report)

    cm = compute_confusion_matrix(labels, preds, len(class_names))
    fig_size = max(8, len(class_names) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusao - {split_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / f"matriz_confusao_{split_name}.png", dpi=150)
    plt.close(fig)

    print(f"\nAcuracia em {split_name}: {acc:.4f}")
    print(report)
    return acc


def dry_run_forward(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)

    print("\nDry-run concluido")
    print(f"Batch de entrada : {tuple(images.shape)}")
    print(f"Saida do modelo  : {tuple(logits.shape)}")
    print(f"Loss do batch    : {loss.item():.4f}")


def load_initial_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    class_names: List[str],
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint inicial nao encontrado: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_classes = checkpoint.get("classes")
    if checkpoint_classes is not None and list(checkpoint_classes) != list(class_names):
        raise RuntimeError(
            "As classes do checkpoint inicial nao batem com o dataset atual.\n"
            f"checkpoint: {checkpoint_classes}\n"
            f"dataset   : {class_names}"
        )

    model.load_state_dict(checkpoint["model_state"])
    print(f"Checkpoint inicial carregado: {checkpoint_path}")


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.prepare_augmented_data:
        config.data_dir = prepare_augmented_dataset(config)
        if config.augment_only:
            print("Modo augment-only: dataset aumentado criado, treino nao iniciado.")
            return

    device, device_label = resolve_device(config.device)
    use_amp = config.amp and device_label == "cuda"

    print(f"Dispositivo: {device} ({device_label})")
    print(f"Dataset    : {config.data_dir.resolve()}")
    print(f"Resultados : {config.output_dir.resolve()}")

    datasets_by_split = load_datasets(config.data_dir, config.image_size)
    class_names = datasets_by_split["train"].classes
    save_dataset_summary(datasets_by_split, config.output_dir)
    print_dataset_summary(datasets_by_split)

    loaders = make_loaders(datasets_by_split, config, device_label)
    model = build_model(
        num_classes=len(class_names),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout,
        device=device,
    )
    if config.initial_checkpoint is not None:
        load_initial_checkpoint(model, config.initial_checkpoint, class_names)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Parametros totais    : {total_params:,}")
    print(f"Parametros treinaveis: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs),
        eta_min=config.min_learning_rate,
    )
    scaler = make_grad_scaler(use_amp)

    if config.dry_run:
        dry_run_forward(model, loaders["train"], criterion, device)
        return

    best_val_acc = -1.0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    checkpoint_path = config.output_dir / config.checkpoint_name
    last_checkpoint_path = config.output_dir / "resnet18_soja_last.pth"
    interrupted_checkpoint_path = config.output_dir / "resnet18_soja_interrupted.pth"
    deadline_ts, deadline_text = resolve_training_deadline(config)
    stopped_by_time = False

    print("\nIniciando treinamento")
    if deadline_text:
        print(f"Horario limite: {deadline_text}")
    print("-" * 72)
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        if deadline_reached(deadline_ts):
            print("Horario limite atingido antes de iniciar nova epoca.")
            stopped_by_time = True
            break

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoca {epoch:03d}/{config.epochs:03d} | lr={current_lr:.8f}")

        train_loss, train_acc, interrupted = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            deadline_ts=deadline_ts,
        )
        if interrupted:
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    class_names=class_names,
                    config=config,
                    history=history,
                ),
                interrupted_checkpoint_path,
            )
            print(
                "Horario limite atingido durante a epoca. "
                f"Checkpoint parcial salvo: {interrupted_checkpoint_path}"
            )
            stopped_by_time = True
            break

        if deadline_reached(deadline_ts):
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    class_names=class_names,
                    config=config,
                    history=history,
                ),
                interrupted_checkpoint_path,
            )
            print(
                "Horario limite atingido antes da validacao. "
                f"Checkpoint parcial salvo: {interrupted_checkpoint_path}"
            )
            stopped_by_time = True
            break

        val_loss, val_acc = evaluate_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            split_name="val",
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        save_history_csv(history, config.output_dir)
        plot_history(history, config.output_dir)

        print(
            f"Treino loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    class_names=class_names,
                    config=config,
                    history=history,
                ),
                checkpoint_path,
            )
            epochs_without_improvement = 0
            print(f"Novo melhor checkpoint salvo: {checkpoint_path} | val_acc={val_acc:.4f}")
        else:
            epochs_without_improvement += 1

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                class_names=class_names,
                config=config,
                history=history,
            ),
            last_checkpoint_path,
        )

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            print(
                f"Early stopping: sem melhora em val_acc por "
                f"{config.patience} epocas."
            )
            break

    elapsed_min = (time.time() - start) / 60
    print("-" * 72)
    print(f"Treinamento concluido em {elapsed_min:.1f} min.")
    print(f"Melhor val_acc: {best_val_acc:.4f}")

    if stopped_by_time and deadline_reached(deadline_ts):
        print("Relatorio final pulado para respeitar o horario limite.")
        print("\nArquivos gerados:")
        print(f"  Melhor modelo : {checkpoint_path}")
        print(f"  Ultimo modelo : {last_checkpoint_path}")
        print(f"  Checkpoint parcial: {interrupted_checkpoint_path}")
        print(f"  Historico     : {config.output_dir / 'historico_treinamento.csv'}")
        print(f"  Curvas        : {config.output_dir / 'curvas_treinamento.png'}")
        print(f"  Classes       : {config.output_dir / 'classes.json'}")
        return

    if best_val_acc < 0:
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                best_val_acc=best_val_acc,
                class_names=class_names,
                config=config,
                history=history,
            ),
            interrupted_checkpoint_path,
        )
        print(f"Nenhuma epoca completa. Checkpoint salvo: {interrupted_checkpoint_path}")
        return

    model.load_state_dict(best_model_state)
    eval_split = "test" if "test" in loaders else "val"
    save_final_report(
        model=model,
        loader=loaders[eval_split],
        class_names=class_names,
        output_dir=config.output_dir,
        device=device,
        split_name=eval_split,
    )

    print("\nArquivos gerados:")
    print(f"  Melhor modelo : {checkpoint_path}")
    print(f"  Ultimo modelo : {last_checkpoint_path}")
    print(f"  Historico     : {config.output_dir / 'historico_treinamento.csv'}")
    print(f"  Curvas        : {config.output_dir / 'curvas_treinamento.png'}")
    print(f"  Classes       : {config.output_dir / 'classes.json'}")


def load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, List[str]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["classes"]
    config = checkpoint.get("config", {})

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
        dropout=float(config.get("dropout", 0.35)),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names


@torch.no_grad()
def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    image_size: int = 224,
    device_name: str = "auto",
) -> Tuple[str, float, Dict[str, float]]:
    device, _ = resolve_device(device_name)
    model, class_names = load_model_for_inference(checkpoint_path, device)
    eval_transform = build_transforms(image_size)["val"]

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    best_idx = int(np.argmax(probs))
    probabilities = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probs)
    }
    return class_names[best_idx], float(probs[best_idx]), probabilities


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
