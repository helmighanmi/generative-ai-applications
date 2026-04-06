"""
Streamlit UI for pplx-embed-v1 local embedding service.
Connects to the FastAPI backend running on localhost:8000.
"""

import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os
from pathlib import Path
from typing import List

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.export_projector import EmbeddingProjector

# ─── Config ─────────────────────────────────────────────────────────────────

API_URL = "http://api:8000"   # Docker service name; override with env var if needed
import os
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="pplx-embed • Embeddings locaux",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg: #0d0f14;
    --surface: #161920;
    --surface2: #1e2130;
    --accent: #7c6af7;
    --accent2: #46e8b0;
    --warn: #f7c46a;
    --text: #e8eaf0;
    --muted: #6b7280;
    --border: #252837;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background: var(--bg);
  }

  .stApp { background: var(--bg); }

  .main-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #161920 60%, #0d1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -20%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(124,106,247,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
  }

  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 6px;
  }
  .badge-purple { background: rgba(124,106,247,0.15); color: var(--accent); border: 1px solid rgba(124,106,247,0.3); }
  .badge-green  { background: rgba(70,232,176,0.12); color: var(--accent2); border: 1px solid rgba(70,232,176,0.25); }
  .badge-yellow { background: rgba(247,196,106,0.12); color: var(--warn); border: 1px solid rgba(247,196,106,0.25); }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.6rem;
  }

  .metric-big {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
  }

  .embed-vector {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent2);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    overflow-x: auto;
    white-space: nowrap;
    line-height: 1.6;
  }

  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }
  .status-ok   { background: var(--accent2); box-shadow: 0 0 6px rgba(70,232,176,0.6); }
  .status-err  { background: #f87171; box-shadow: 0 0 6px rgba(248,113,113,0.6); }
  .status-warn { background: var(--warn); box-shadow: 0 0 6px rgba(247,196,106,0.5); }

  div[data-testid="stTextArea"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
  }
  div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b4de8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,106,247,0.4) !important;
  }

  .stSlider [data-testid="stSlider"] { color: var(--accent) !important; }

  [data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
  }

  .sidebar-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  hr { border-color: var(--border) !important; opacity: 0.5; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_embed(texts: List[str], batch_size: int = 16):
    r = requests.post(
        f"{API_URL}/embed",
        json={"texts": texts, "batch_size": batch_size},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def api_similarity(a: str, b: str):
    r = requests.post(
        f"{API_URL}/similarity",
        json={"text_a": a, "text_b": b},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def cosine_matrix(embeddings: List[List[float]]) -> np.ndarray:
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = arr / norms
    return normed @ normed.T


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#6b7280; letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;">
      Configuration
    </div>
    """, unsafe_allow_html=True)

    health = api_health()
    if health and health.get("model_loaded"):
        st.markdown('<span class="status-dot status-ok"></span> **API connectée**', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-dot status-err"></span> **API non disponible**', unsafe_allow_html=True)
        st.caption(f"URL: `{API_URL}`")

    st.markdown("---")

    batch_size = st.slider("Batch size", min_value=1, max_value=64, value=16, step=1,
                           help="Nombre de textes traités en parallèle (plus grand = plus rapide si RAM suffisante)")

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.5rem;">
      Modèle
    </div>
    """, unsafe_allow_html=True)
    st.code("pplx-embed-v1-0.6B", language=None)
    st.markdown('<span class="badge badge-purple">CPU</span><span class="badge badge-green">Local</span><span class="badge badge-yellow">FR</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#6b7280;">
    📚 Optimisé pour le français<br>
    🔒 100% local, aucune API externe<br>
    📐 1024 dimensions · INT8
    </div>
    """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🧠 pplx-embed · Studio</h1>
  <p>Calcul d'embeddings locaux avec <strong>pplx-embed-v1-0.6B</strong> de Perplexity AI — aucune donnée ne quitte votre machine.</p>
  <br>
  <span class="badge badge-purple">MIT License</span>
  <span class="badge badge-green">100% Local</span>
  <span class="badge badge-yellow">Multilingual · FR</span>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📐 Embeddings", "🔁 Similarité", "📊 Matrice", "🔭 Projecteur"])


# ═══ Tab 1 — Embed texts ══════════════════════════════════════════════════════
with tab1:
    st.markdown("### Calculer des embeddings")
    st.caption("Entrez vos textes (un par ligne). Idéal pour documents RAG en français.")

    col_input, col_out = st.columns([1, 1], gap="large")

    with col_input:
        raw_input = st.text_area(
            "Textes à encoder",
            height=220,
            placeholder="La photosynthèse est le processus par lequel les plantes...\nLe changement climatique affecte les écosystèmes...\nLes modèles de langage transforment l'IA...",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            run_btn = st.button("⚡ Calculer embeddings", use_container_width=True)
        with c2:
            show_raw = st.checkbox("Afficher vecteurs bruts", value=False)

    with col_out:
        if run_btn:
            texts = [t.strip() for t in raw_input.strip().split("\n") if t.strip()]
            if not texts:
                st.warning("⚠️ Veuillez entrer au moins un texte.")
            elif health is None or not health.get("model_loaded"):
                st.error("❌ L'API n'est pas disponible. Vérifiez que le service est démarré.")
            else:
                with st.spinner("Calcul en cours sur CPU..."):
                    try:
                        result = api_embed(texts, batch_size=batch_size)
                        st.session_state["last_embed_result"] = result
                        st.session_state["last_embed_texts"] = texts
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        result = None

        result = st.session_state.get("last_embed_result")
        texts_used = st.session_state.get("last_embed_texts", [])

        if result:
            m1, m2, m3 = st.columns(3)
            m1.metric("Textes", result["num_texts"])
            m2.metric("Dimensions", result["dimensions"])
            m3.metric("Temps", f"{result['processing_time_ms']:.0f} ms")

            if show_raw:
                for i, (txt, emb) in enumerate(zip(texts_used, result["embeddings"])):
                    short = txt[:60] + "..." if len(txt) > 60 else txt
                    with st.expander(f"[{i+1}] {short}"):
                        preview = emb[:24]
                        st.markdown(
                            f'<div class="embed-vector">[{", ".join(f"{v:.4f}" for v in preview)}, ...]</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"Vecteur de {len(emb)} dimensions · norme={np.linalg.norm(emb):.4f}")

            # Download button
            dl_data = json.dumps({
                "model": result["model"],
                "texts": texts_used,
                "embeddings": result["embeddings"],
            }, ensure_ascii=False, indent=2)
            st.download_button(
                "💾 Télécharger JSON",
                data=dl_data,
                file_name="embeddings.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1rem; color:#6b7280;">
              <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔮</div>
              <div style="font-family:'Space Mono',monospace; font-size:0.8rem;">
                Les vecteurs apparaîtront ici
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══ Tab 2 — Similarity ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Comparer deux textes")
    st.caption("Mesure la similarité cosinus entre deux phrases ou paragraphes.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        text_a = st.text_area("Texte A", height=130,
                              placeholder="Le chat mange du poisson.",
                              label_visibility="visible")
    with col_b:
        text_b = st.text_area("Texte B", height=130,
                              placeholder="Le félin se nourrit de poisson.",
                              label_visibility="visible")

    if st.button("🔁 Calculer la similarité", use_container_width=False):
        if not text_a.strip() or not text_b.strip():
            st.warning("Remplissez les deux champs.")
        elif health is None or not health.get("model_loaded"):
            st.error("❌ API non disponible.")
        else:
            with st.spinner("Calcul..."):
                try:
                    res = api_similarity(text_a.strip(), text_b.strip())
                    score = res["cosine_similarity"]
                    st.session_state["sim_score"] = score
                    st.session_state["sim_texts"] = (text_a.strip(), text_b.strip())
                except Exception as e:
                    st.error(f"Erreur : {e}")

    sim_score = st.session_state.get("sim_score")
    if sim_score is not None:
        col_score, col_gauge = st.columns([1, 2], gap="large")
        with col_score:
            color = "#46e8b0" if sim_score > 0.7 else "#f7c46a" if sim_score > 0.4 else "#f87171"
            label = "Très similaires" if sim_score > 0.7 else "Assez similaires" if sim_score > 0.4 else "Peu similaires"
            st.markdown(f"""
            <div class="card" style="text-align:center;">
              <div class="card-title">Similarité cosinus</div>
              <div class="metric-big" style="color:{color}">{sim_score:.4f}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sim_score,
                number={"font": {"color": color, "size": 36}},
                gauge={
                    "axis": {"range": [-1, 1], "tickcolor": "#6b7280"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#1e2130",
                    "bordercolor": "#252837",
                    "steps": [
                        {"range": [-1, 0.4], "color": "#1a1d2e"},
                        {"range": [0.4, 0.7], "color": "#1f2535"},
                        {"range": [0.7, 1.0], "color": "#1a2d28"},
                    ],
                },
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e8eaf0",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══ Tab 3 — Similarity matrix ════════════════════════════════════════════════════
with tab3:
    st.markdown("### Visualiser la matrice de similarité")
    st.caption("Calculez les embeddings dans l'onglet **Embeddings** puis visualisez les relations entre vos textes.")

    result = st.session_state.get("last_embed_result")
    texts_used = st.session_state.get("last_embed_texts", [])

    if result and len(result["embeddings"]) >= 2:
        matrix = cosine_matrix(result["embeddings"])
        labels = [t[:40] + "…" if len(t) > 40 else t for t in texts_used]

        fig = px.imshow(
            matrix,
            x=labels, y=labels,
            color_continuous_scale=[[0, "#0d0f14"], [0.5, "#7c6af7"], [1, "#46e8b0"]],
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#161920",
            font_color="#e8eaf0",
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            coloraxis_colorbar=dict(tickfont=dict(color="#6b7280")),
            margin=dict(l=20, r=20, t=30, b=20),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        n = len(texts_used)
        upper = matrix[np.triu_indices(n, k=1)]
        if len(upper) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Sim. max", f"{upper.max():.4f}")
            c2.metric("Sim. moyenne", f"{upper.mean():.4f}")
            c3.metric("Sim. min", f"{upper.min():.4f}")

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 1rem; color:#6b7280;">
          <div style="font-size:2.5rem; margin-bottom:0.8rem;">📊</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.8rem;">
            Calculez d'abord des embeddings (≥2 textes)<br>dans l'onglet Embeddings
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══ Tab 4 — Embedding Projector ══════════════════════════════════════════════
with tab4:
    st.markdown("### 🔭 Projecteur d'embeddings")
    st.caption("Visualisation 2D/3D de l'espace sémantique · PCA, UMAP ou t-SNE · Export HTML autonome")

    result = st.session_state.get("last_embed_result")
    texts_used = st.session_state.get("last_embed_texts", [])

    col_cfg, col_main = st.columns([1, 3], gap="large")

    with col_cfg:
        st.markdown('<div class="card-title">Configuration</div>', unsafe_allow_html=True)
        proj_method = st.selectbox("Méthode de réduction", ["umap", "pca", "tsne"],
                                   format_func=lambda x: x.upper())
        proj_dim = st.radio("Dimensions", ["2D", "3D"], horizontal=True)
        n_clusters = st.slider("Clusters (K-Means)", 2, 12, 5)

        st.markdown("---")
        show_var = st.checkbox("Afficher scree plot PCA", value=True)

        st.markdown("---")
        run_proj = st.button("🔭 Projeter", use_container_width=True)

        if result:
            st.markdown("---")
            export_btn = st.button("💾 Exporter HTML", use_container_width=True)
        else:
            export_btn = False

    with col_main:
        if not result or len(result.get("embeddings", [])) < 2:
            st.markdown("""
            <div class="card" style="text-align:center; padding:4rem 1rem; color:#6b7280;">
              <div style="font-size:3rem; margin-bottom:1rem;">🔭</div>
              <div style="font-family:'Space Mono',monospace; font-size:0.8rem;">
                Calculez d'abord des embeddings (≥2 textes)<br>dans l'onglet <strong>Embeddings</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            embeddings = result["embeddings"]
            labels = texts_used

            if run_proj or "proj_fig" not in st.session_state:
                with st.spinner(f"Réduction {proj_method.upper()} en cours..."):
                    try:
                        proj = EmbeddingProjector(embeddings, labels=labels)
                        if proj_dim == "2D":
                            fig = proj.plot(method=proj_method, n_clusters=n_clusters,
                                            title="Embedding Projector")
                        else:
                            fig = proj.plot3d(method=proj_method, n_clusters=n_clusters,
                                              title="Embedding Projector 3D")
                        st.session_state["proj_fig"] = fig
                        st.session_state["proj_obj"] = proj
                        st.session_state["proj_method"] = proj_method
                    except Exception as e:
                        st.error(f"Erreur projection : {e}")

            proj_fig = st.session_state.get("proj_fig")
            if proj_fig:
                st.plotly_chart(proj_fig, use_container_width=True)

            # Scree plot
            if show_var:
                proj_obj = st.session_state.get("proj_obj")
                if proj_obj:
                    with st.expander("📈 Scree Plot PCA — Variance expliquée", expanded=False):
                        fig_var = proj_obj.plot_variance()
                        st.plotly_chart(fig_var, use_container_width=True)

            # Export HTML
            if export_btn:
                proj_obj = st.session_state.get("proj_obj")
                if proj_obj:
                    with st.spinner("Export HTML en cours..."):
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                            proj_obj.export_html(
                                output_path=tmp.name,
                                method=st.session_state.get("proj_method", "pca"),
                                n_clusters=n_clusters,
                                title="Embedding Projector — pplx-embed-v1",
                            )
                            html_bytes = Path(tmp.name).read_bytes()

                    st.download_button(
                        "⬇️ Télécharger projector.html",
                        data=html_bytes,
                        file_name="projector.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                    st.success("✅ Fichier HTML autonome prêt — ouvrez-le sans serveur !")
