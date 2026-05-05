# API ResNet18 Soja no Render

Esta API usa FastAPI para classificar imagens de folhas de soja com o checkpoint treinado.

## Arquivos principais

- `main.py`: API de inferencia.
- `requirements.txt`: dependencias do Render.
- `render.yaml`: configuracao de deploy.
- `resnet18_soja.py`: script de treino.

## Endpoints

- `GET /health`: status da API e do modelo.
- `GET /classes`: lista das classes.
- `POST /predict`: recebe uma imagem via `multipart/form-data`.
- `POST /predict-base64`: recebe imagem em base64.
- `GET /docs`: Swagger UI automatico do FastAPI.

## Rodar localmente

```powershell
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload
```

Abra:

```text
http://127.0.0.1:8000/docs
```

## Testar localmente via PowerShell

```powershell
$img = "caminho\para\imagem.jpg"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict?top_k=3" -Method Post -Form @{ file = Get-Item $img }
```

## Deploy no Render

No Render, crie um Web Service apontando para o repositorio.

Config padrao:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

O arquivo `.pth` tem mais de 100 MB, entao normalmente voce nao deve subir ele direto no GitHub.
Suba o checkpoint em algum local com URL direta de download e configure no Render:

```text
MODEL_URL=https://sua-url-direta/resnet18_soja_best.pth
```

A API baixa esse arquivo na inicializacao e carrega o modelo.

## Checkpoint recomendado

O melhor checkpoint local foi:

```text
resultados_100ep_aug/resnet18_soja_best.pth
```

Ele teve melhor validacao geral do que o treino posterior com `data_augmented_max`.
