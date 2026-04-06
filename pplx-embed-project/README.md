# 🧠 pplx-embed — Local Embedding Stack

> Embeddings 100% locaux avec **[pplx-embed-v1-0.6B](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)** de Perplexity AI.
> Aucune donnée ne quitte votre machine. Fonctionne sur CPU.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🔒 **100% local** | Aucune API externe, aucune donnée envoyée |
| 🖥️ **CPU only** | Pas de GPU requis |
| 🌍 **Multilingue** | Optimisé pour le français et 100+ langues |
| ⚡ **API REST** | FastAPI avec `/embed`, `/similarity`, `/health` |
| 🎨 **Interface Streamlit** | 4 onglets : Embeddings · Similarité · Matrice · Projecteur |
| 🔭 **Embedding Projector** | PCA / UMAP / t-SNE · 2D/3D · Export HTML autonome |
| 🐳 **Docker** | Stack complète avec `docker compose up` |
| 📦 **Installable** | `pip install -e .[dev]` via `pyproject.toml` |

## 🐍 Python & Virtual Environment

### ✅ Recommended Python version

This project works with:

- Python 3.10 / 3.11 (recommended)
- Python 3.12+ may cause compatibility issues with some dependencies (e.g. PyTorch)

Check your version: python --version

---

## 📁 Structure du projet

```
pplx-embed-project/
│
├── api/
│   ├── main.py          ← Endpoints : /embed /similarity /health
│   ├── embedder.py      ← Wrapper modèle (sentence-transformers, CPU)
│   └── requirements.txt
│
├── streamlit_app/
│   ├── app.py           ← Interface Streamlit (4 onglets)
│   └── requirements.txt
│
├── scripts/
│   ├── download_model.py    ← Pré-télécharge les poids du modèle
│   └── export_projector.py  ← Projecteur PCA/UMAP/t-SNE + export HTML
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.streamlit
│
├── model/               ← Cache des poids du modèle (auto-créé)
├── launch.sh            ← Launcher unifié
├── pyproject.toml       ← Package Python installable
└── docker-compose.yml
```

---

## 🚀 Démarrage rapide

### Option A — Docker (recommandé)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Interface Streamlit | http://localhost:8501 |
| API FastAPI | http://localhost:8000 |
| Docs interactives | http://localhost:8000/docs |

> Premier démarrage : ~5–10 min (téléchargement ~1.2 GB + build).

---

### Option B — Python local (dev)

```bash
./launch.sh install   # crée .venv + pip install -e .[dev]
./launch.sh all       # démarre API + Streamlit en local
./launch.sh stop      # arrête les processus
```

---

## 🛠 launch.sh — Référence

```
Commandes Docker:
  docker         Démarre API + Streamlit (Docker Compose)
  docker:api     Démarre uniquement l'API
  docker:ui      Démarre uniquement Streamlit
  docker:down    Arrête tous les conteneurs
  docker:logs    Suit les logs

Commandes Python local:
  install        Crée .venv et installe le projet
  api            Lance FastAPI avec hot-reload
  ui             Lance Streamlit
  all            Lance les deux en arrière-plan
  stop           Arrête les processus locaux

Outils:
  download       Télécharge les poids du modèle
  projector      Exporte le projecteur en HTML
```

---

## 📦 Installation Python

```bash
pip install -e ".[dev]"        # tout (recommandé pour le développement)
pip install -e ".[cpu]"        # CPU-only torch (plus léger)
pip install -e ".[projector]"  # UMAP + visualisation seulement
pip install -e ".[ui]"         # Streamlit seulement
```

---

## 🔌 API REST

### `GET /health`
```bash
curl http://localhost:8000/health
```

### `POST /embed`
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Bonjour le monde", "La France est belle"], "batch_size": 16}'
```

### `POST /similarity`
```bash
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"text_a": "Le chat mange du poisson.", "text_b": "Le félin se nourrit de poisson."}'
```

---

## 🔭 Embedding Projector

Visualise l'espace sémantique de tes embeddings en 2D/3D avec clustering automatique.

### CLI
```bash
# Démo intégrée (16 phrases françaises, 4 catégories)
./launch.sh projector --demo

# Sur tes propres embeddings
./launch.sh projector --input embeddings.json --method umap --output projector.html
./launch.sh projector --input embeddings.json --method pca --no-3d
./launch.sh projector --input embeddings.json --method tsne --clusters 8
```

### Module Python
```python
from scripts.export_projector import EmbeddingProjector

proj = EmbeddingProjector(embeddings, labels=texts, categories=categories)

fig = proj.plot(method="umap", n_clusters=5)   # figure Plotly interactive
fig.show()

proj.export_html("projector.html", method="umap", include_3d=True)
```

---

## 🐍 Utiliser l'API dans ton projet RAG

```python
import requests, numpy as np

def embed(texts):
    r = requests.post("http://localhost:8000/embed", json={"texts": texts})
    return r.json()["embeddings"]

query = "Quand a commencé la Révolution française ?"
docs  = ["La Révolution a débuté en 1789.", "Napoléon fut sacré en 1804."]

q_emb  = embed([query])[0]
d_embs = embed(docs)

scores = [np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d)) for d in d_embs]
print(docs[np.argmax(scores)])
```

---

## 📊 Performances indicatives (CPU)

| Textes | Temps |
|---|---|
| 1 | ~200 ms |
| 10 | ~800 ms |
| 100 | ~6 s |
| 500 | ~30 s |

---

## 📚 Ressources

- [pplx-embed-v1-0.6B sur HuggingFace](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)
- [Article technique Perplexity](https://research.perplexity.ai/articles/pplx-embed-state-of-the-art-embedding-models-for-web-scale-retrieval)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

## 📄 Licence

MIT
