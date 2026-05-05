"""
API FastAPI para inferencia do modelo ResNet-18 de doencas em folhas de soja.

Endpoints principais:
  GET  /health
  GET  /classes
  POST /predict          multipart/form-data com campo "file"
  POST /predict-base64   JSON com "image_base64"

No Render, configure MODEL_URL com uma URL direta para o arquivo .pth.
Localmente, a API tenta carregar automaticamente o melhor checkpoint existente.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import models, transforms


APP_NAME = "API ResNet18 Soja"
DEFAULT_MODEL_CANDIDATES = [
    Path("model/resnet18_soja_best.pth"),
    Path("resultados_100ep_aug/resnet18_soja_best.pth"),
    Path("resultados_ate_1730_max/resnet18_soja_best.pth"),
    Path("resultados/resnet18_soja_best.pth"),
]


class PredictionItem(BaseModel):
    classe: str
    probabilidade: float
    percentual: float


class PredictionResponse(BaseModel):
    predicted_class: str
    class_name: str
    classe: str
    classe_predita: str
    resultado: str
    doenca: str
    confidence: float
    confidence_percent: float
    confianca: float
    confianca_percentual: float
    percentual_confianca: float
    top_k: List[PredictionItem]
    probabilities: Dict[str, float]


class Base64Request(BaseModel):
    image_base64: str = Field(..., description="Imagem em base64, com ou sem prefixo data URL.")
    top_k: int = Field(3, ge=1, le=10)


class ModelBundle:
    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.classes: List[str] = []
        self.model_path: Optional[Path] = None
        self.load_error: Optional[str] = None
        self.device = torch.device("cpu")

    @property
    def loaded(self) -> bool:
        return self.model is not None and bool(self.classes)


bundle = ModelBundle()

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="Classificacao de doencas em folhas de soja usando ResNet-18.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(num_classes: int, dropout: float = 0.35) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout * 0.75),
        nn.Linear(256, num_classes),
    )
    return model


def torch_load_checkpoint(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_model_path() -> Path:
    model_url = os.getenv("MODEL_URL", "").strip()
    env_model_path = os.getenv("MODEL_PATH", "").strip()

    if model_url:
        target = Path(env_model_path or "/tmp/resnet18_soja_best.pth")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Baixando modelo de MODEL_URL para {target}...")
            urllib.request.urlretrieve(model_url, target)
            print("Download do modelo concluido.")
        return target

    if env_model_path:
        path = Path(env_model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"MODEL_PATH informado, mas arquivo nao existe: {path}")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Nenhum checkpoint encontrado. Defina MODEL_URL no Render ou MODEL_PATH localmente."
    )


def load_model() -> None:
    model_path = resolve_model_path()
    checkpoint = torch_load_checkpoint(model_path)

    classes = checkpoint.get("classes")
    if not classes:
        raise RuntimeError("Checkpoint sem lista de classes.")

    config = checkpoint.get("config", {})
    dropout = float(config.get("dropout", 0.35))

    model = build_model(num_classes=len(classes), dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(bundle.device)
    model.eval()

    bundle.model = model
    bundle.classes = list(classes)
    bundle.model_path = model_path
    bundle.load_error = None


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_model()
    except Exception as exc:
        bundle.load_error = str(exc)
        print(f"Modelo nao carregado no startup: {exc}")


def read_image_from_bytes(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Arquivo enviado nao e uma imagem valida.") from exc


def ensure_model_loaded() -> nn.Module:
    if not bundle.loaded:
        try:
            load_model()
        except Exception as exc:
            bundle.load_error = str(exc)
            raise HTTPException(
                status_code=503,
                detail=f"Modelo nao carregado: {exc}",
            ) from exc
    if bundle.model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado.")
    return bundle.model


def predict_image(image: Image.Image, top_k: int = 3) -> PredictionResponse:
    model = ensure_model_loaded()
    if top_k < 1:
        top_k = 1
    top_k = min(top_k, len(bundle.classes))

    transform = get_transform()
    tensor = transform(image).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    sorted_indices = torch.argsort(probs, descending=True).tolist()
    best_idx = sorted_indices[0]

    top_items = [
        PredictionItem(
            classe=bundle.classes[idx],
            probabilidade=float(probs[idx]),
            percentual=float(probs[idx] * 100),
        )
        for idx in sorted_indices[:top_k]
    ]
    probabilities = {
        class_name: float(probs[idx])
        for idx, class_name in enumerate(bundle.classes)
    }

    predicted_class = bundle.classes[best_idx]
    confidence = float(probs[best_idx])
    confidence_percent = float(probs[best_idx] * 100)

    return PredictionResponse(
        predicted_class=predicted_class,
        class_name=predicted_class,
        classe=predicted_class,
        classe_predita=predicted_class,
        resultado=predicted_class,
        doenca=predicted_class,
        confidence=confidence,
        confidence_percent=confidence_percent,
        confianca=confidence_percent,
        confianca_percentual=confidence_percent,
        percentual_confianca=confidence_percent,
        top_k=top_items,
        probabilities=probabilities,
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": APP_NAME,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok" if bundle.loaded else "model_not_loaded",
        "model_loaded": bundle.loaded,
        "model_path": str(bundle.model_path) if bundle.model_path else None,
        "load_error": bundle.load_error,
        "classes": len(bundle.classes),
    }


@app.get("/classes")
def classes() -> Dict[str, object]:
    ensure_model_loaded()
    return {"total": len(bundle.classes), "classes": bundle.classes}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10),
) -> PredictionResponse:
    raw = await file.read()
    image = read_image_from_bytes(raw)
    return predict_image(image, top_k=top_k)


@app.post("/predict-base64", response_model=PredictionResponse)
def predict_base64(payload: Base64Request) -> PredictionResponse:
    raw_text = payload.image_base64.strip()
    if "," in raw_text and raw_text.lower().startswith("data:"):
        raw_text = raw_text.split(",", 1)[1]

    try:
        raw = base64.b64decode(raw_text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Base64 invalido.") from exc

    image = read_image_from_bytes(raw)
    return predict_image(image, top_k=payload.top_k)
