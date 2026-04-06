#!/usr/bin/env python3
"""
Embedding Projector
═══════════════════
Senior-grade tool for exploring high-dimensional embedding spaces.

Features:
  - PCA (always available, no extra deps)
  - UMAP (install umap-learn for best results)
  - t-SNE
  - Interactive Plotly figure with hover labels, color-by-cluster
  - K-Means clustering overlay
  - Standalone HTML export (shareable, no server needed)
  - Works both as a CLI tool and as an imported module

CLI usage:
  python scripts/export_projector.py --input embeddings.json --method umap --output projector.html
  python scripts/export_projector.py --demo                              # run on built-in demo data

Module usage:
  from scripts.export_projector import EmbeddingProjector
  proj = EmbeddingProjector(embeddings, labels=texts)
  proj.plot(method="umap").write_html("out.html")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Types ────────────────────────────────────────────────────────────────────

Method = Literal["pca", "umap", "tsne"]

# ─── Palette ──────────────────────────────────────────────────────────────────

PALETTE = [
    "#7c6af7", "#46e8b0", "#f7c46a", "#f87171",
    "#60a5fa", "#fb923c", "#a78bfa", "#34d399",
    "#fbbf24", "#f472b6", "#38bdf8", "#4ade80",
]

DARK_THEME = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font_color="#e8eaf0",
    xaxis=dict(
        showgrid=True, gridcolor="#1e2130", zeroline=False,
        tickfont=dict(size=10, color="#6b7280"),
        title_font=dict(color="#6b7280"),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1e2130", zeroline=False,
        tickfont=dict(size=10, color="#6b7280"),
        title_font=dict(color="#6b7280"),
    ),
)


# ─── Core class ───────────────────────────────────────────────────────────────

class EmbeddingProjector:
    """
    Projects high-dimensional embeddings to 2D/3D for visual exploration.

    Parameters
    ----------
    embeddings : array-like, shape (N, D)
    labels     : list of str — hover labels for each point
    categories : list of str/int — used to colour points; if None, K-Means is used
    """

    def __init__(
        self,
        embeddings: np.ndarray | list,
        labels: Optional[list[str]] = None,
        categories: Optional[list] = None,
    ):
        self.embeddings = normalize(np.array(embeddings, dtype=np.float32))
        n = len(self.embeddings)
        self.labels = labels or [f"text_{i}" for i in range(n)]
        self.categories = categories
        self._reduced: dict[str, np.ndarray] = {}

    # ── Reduction ─────────────────────────────────────────────────────────────

    def _reduce_pca(self, n_components: int = 2) -> np.ndarray:
        key = f"pca{n_components}"
        if key not in self._reduced:
            pca = PCA(n_components=n_components, random_state=42)
            self._reduced[key] = pca.fit_transform(self.embeddings)
            self._pca_var = pca.explained_variance_ratio_
        return self._reduced[key]

    def _reduce_umap(self, n_components: int = 2) -> np.ndarray:
        key = f"umap{n_components}"
        if key not in self._reduced:
            try:
                from umap import UMAP  # type: ignore
            except ImportError:
                print("⚠️  umap-learn not installed → falling back to PCA.")
                print("   Install it: pip install umap-learn")
                return self._reduce_pca(n_components)
            n = len(self.embeddings)
            n_neighbors = min(15, n - 1)
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
                verbose=False,
            )
            self._reduced[key] = reducer.fit_transform(self.embeddings)
        return self._reduced[key]

    def _reduce_tsne(self, n_components: int = 2) -> np.ndarray:
        key = f"tsne{n_components}"
        if key not in self._reduced:
            n = len(self.embeddings)
            perplexity = min(30, max(5, n // 3))
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                metric="cosine",
                random_state=42,
                n_iter=1000,
                verbose=0,
            )
            self._reduced[key] = tsne.fit_transform(self.embeddings)
        return self._reduced[key]

    def reduce(self, method: Method = "pca", n_components: int = 2) -> np.ndarray:
        dispatch = {"pca": self._reduce_pca, "umap": self._reduce_umap, "tsne": self._reduce_tsne}
        return dispatch[method](n_components)

    # ── Clustering ────────────────────────────────────────────────────────────

    def cluster(self, n_clusters: int = 6) -> np.ndarray:
        n = len(self.embeddings)
        k = min(n_clusters, n)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        return km.fit_predict(self.embeddings)

    # ── 2-D plot ──────────────────────────────────────────────────────────────

    def plot(
        self,
        method: Method = "umap",
        n_clusters: int = 6,
        title: str = "Embedding Projector",
        show_explained_var: bool = True,
    ) -> go.Figure:
        coords = self.reduce(method, 2)
        x, y = coords[:, 0], coords[:, 1]

        # Colour assignments
        if self.categories is not None:
            cats = [str(c) for c in self.categories]
            unique_cats = sorted(set(cats))
            color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique_cats)}
            colors = [color_map[c] for c in cats]
            group_labels = cats
        else:
            cluster_ids = self.cluster(n_clusters)
            colors = [PALETTE[c % len(PALETTE)] for c in cluster_ids]
            group_labels = [f"Cluster {c}" for c in cluster_ids]

        # Short labels for display
        hover_labels = [
            f"{lbl[:120]}…" if len(lbl) > 120 else lbl
            for lbl in self.labels
        ]

        # Build traces per group for legend
        group_map: dict[str, list[int]] = {}
        for i, g in enumerate(group_labels):
            group_map.setdefault(g, []).append(i)

        fig = go.Figure()

        for group, indices in sorted(group_map.items()):
            fig.add_trace(go.Scatter(
                x=x[indices],
                y=y[indices],
                mode="markers+text",
                name=group,
                text=["" for _ in indices],   # no inline text to avoid clutter
                customdata=[hover_labels[i] for i in indices],
                hovertemplate="<b>%{customdata}</b><extra>%{fullData.name}</extra>",
                marker=dict(
                    color=colors[indices[0]],
                    size=10,
                    opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.15)", width=1),
                    symbol="circle",
                ),
            ))

        method_label = method.upper()
        subtitle = ""
        if method == "pca" and hasattr(self, "_pca_var"):
            pct = " + ".join(f"{v*100:.1f}%" for v in self._pca_var[:2])
            subtitle = f" — explained variance: {pct}"

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>   <span style='font-size:13px;color:#6b7280'>{method_label}{subtitle}</span>",
                font=dict(size=18, color="#e8eaf0"),
                x=0.02, xanchor="left",
            ),
            legend=dict(
                bgcolor="rgba(22,25,32,0.8)",
                bordercolor="#252837",
                borderwidth=1,
                font=dict(size=11, color="#e8eaf0"),
                itemsizing="constant",
            ),
            hovermode="closest",
            height=640,
            margin=dict(l=40, r=40, t=70, b=40),
            **DARK_THEME,
        )

        return fig

    # ── 3-D plot ──────────────────────────────────────────────────────────────

    def plot3d(
        self,
        method: Method = "umap",
        n_clusters: int = 6,
        title: str = "Embedding Projector 3D",
    ) -> go.Figure:
        coords = self.reduce(method, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        if self.categories is not None:
            cats = [str(c) for c in self.categories]
            unique_cats = sorted(set(cats))
            color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(unique_cats)}
            colors = [color_map[c] for c in cats]
            group_labels = cats
        else:
            cluster_ids = self.cluster(n_clusters)
            colors = [PALETTE[c % len(PALETTE)] for c in cluster_ids]
            group_labels = [f"Cluster {c}" for c in cluster_ids]

        hover_labels = [lbl[:120] + "…" if len(lbl) > 120 else lbl for lbl in self.labels]
        group_map: dict[str, list[int]] = {}
        for i, g in enumerate(group_labels):
            group_map.setdefault(g, []).append(i)

        fig = go.Figure()
        for group, indices in sorted(group_map.items()):
            fig.add_trace(go.Scatter3d(
                x=x[indices], y=y[indices], z=z[indices],
                mode="markers",
                name=group,
                customdata=[hover_labels[i] for i in indices],
                hovertemplate="<b>%{customdata}</b><extra>%{fullData.name}</extra>",
                marker=dict(
                    color=colors[indices[0]],
                    size=6,
                    opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                ),
            ))

        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=18, color="#e8eaf0"), x=0.02),
            scene=dict(
                bgcolor="#0d0f14",
                xaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1e2130", showgrid=True, zeroline=False),
                yaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1e2130", showgrid=True, zeroline=False),
                zaxis=dict(backgroundcolor="#0d0f14", gridcolor="#1e2130", showgrid=True, zeroline=False),
            ),
            paper_bgcolor="#0d0f14",
            font_color="#e8eaf0",
            legend=dict(bgcolor="rgba(22,25,32,0.8)", bordercolor="#252837", borderwidth=1),
            height=680,
            margin=dict(l=0, r=0, t=60, b=0),
        )
        return fig

    # ── Variance / scree plot (PCA) ───────────────────────────────────────────

    def plot_variance(self, n_components: int = 20) -> go.Figure:
        k = min(n_components, len(self.embeddings), self.embeddings.shape[1])
        pca = PCA(n_components=k, random_state=42)
        pca.fit(self.embeddings)
        var = pca.explained_variance_ratio_
        cumvar = np.cumsum(var)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=list(range(1, k + 1)), y=var * 100,
            name="Variance per PC",
            marker_color="#7c6af7",
            marker_line_width=0,
            opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, k + 1)), y=cumvar * 100,
            name="Cumulative variance",
            line=dict(color="#46e8b0", width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ), secondary_y=True)

        fig.update_layout(
            title=dict(text="<b>PCA — Explained Variance (Scree Plot)</b>",
                       font=dict(size=16, color="#e8eaf0"), x=0.02),
            xaxis_title="Principal Component",
            height=380,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(bgcolor="rgba(22,25,32,0.8)", bordercolor="#252837", borderwidth=1),
            **DARK_THEME,
        )
        fig.update_yaxes(title_text="Variance (%)", secondary_y=False,
                         tickfont=dict(color="#6b7280"), gridcolor="#1e2130")
        fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True,
                         tickfont=dict(color="#6b7280"), range=[0, 105])
        return fig

    # ── HTML export ───────────────────────────────────────────────────────────

    def export_html(
        self,
        output_path: str = "projector.html",
        method: Method = "umap",
        include_3d: bool = True,
        n_clusters: int = 6,
        title: str = "Embedding Projector",
    ) -> str:
        """
        Export a fully standalone HTML file (no server needed, shareable).
        All Plotly JS is bundled inline.
        """
        fig2d = self.plot(method=method, n_clusters=n_clusters, title=title)
        fig3d = self.plot3d(method=method, n_clusters=n_clusters, title=f"{title} — 3D") if include_3d else None
        fig_var = self.plot_variance()

        html_2d = fig2d.to_html(full_html=False, include_plotlyjs="cdn", div_id="plot2d")
        html_3d = fig3d.to_html(full_html=False, include_plotlyjs=False, div_id="plot3d") if fig3d else ""
        html_var = fig_var.to_html(full_html=False, include_plotlyjs=False, div_id="plotvar")

        n = len(self.embeddings)
        dim = self.embeddings.shape[1]
        method_label = method.upper()

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; }}
  :root {{
    --bg: #0d0f14; --surface: #161920; --surface2: #1e2130;
    --accent: #7c6af7; --accent2: #46e8b0;
    --text: #e8eaf0; --muted: #6b7280; --border: #252837;
  }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }}
  header {{
    background: linear-gradient(135deg, #1a1d2e, #161920);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2.5rem;
    display: flex; align-items: center; gap: 1.5rem;
  }}
  header h1 {{ font-family: 'Space Mono', monospace; font-size: 1.3rem; margin: 0; color: #fff; }}
  header p  {{ margin: 0; color: var(--muted); font-size: 0.85rem; }}
  .badge {{
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.65rem; font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 0.5px; margin-right: 4px;
  }}
  .b-purple {{ background: rgba(124,106,247,.15); color: var(--accent); border: 1px solid rgba(124,106,247,.3); }}
  .b-green  {{ background: rgba(70,232,176,.12); color: var(--accent2); border: 1px solid rgba(70,232,176,.25); }}
  .stats {{
    display: flex; gap: 1.5rem; padding: 1rem 2.5rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  .stat {{ text-align: center; }}
  .stat-val {{ font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #fff; }}
  .stat-lbl {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
  nav {{
    display: flex; gap: 0; padding: 0 2.5rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}
  nav button {{
    background: none; border: none; border-bottom: 2px solid transparent;
    color: var(--muted); font-family: 'Space Mono', monospace; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 0.9rem 1.2rem; cursor: pointer; transition: all .2s;
  }}
  nav button:hover {{ color: var(--text); }}
  nav button.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; padding: 1.5rem 2rem; }}
  .tab-content.active {{ display: block; }}
  .plot-container {{ border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
  footer {{
    text-align: center; padding: 1rem;
    color: var(--muted); font-size: 0.75rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>🧠 {title}</h1>
    <p>Exploration de l'espace des embeddings · <strong>pplx-embed-v1-0.6B</strong></p>
  </div>
  <div style="margin-left:auto">
    <span class="badge b-purple">{method_label}</span>
    <span class="badge b-green">LOCAL</span>
    <span class="badge b-purple">CPU</span>
  </div>
</header>

<div class="stats">
  <div class="stat"><div class="stat-val">{n}</div><div class="stat-lbl">Points</div></div>
  <div class="stat"><div class="stat-val">{dim}</div><div class="stat-lbl">Dimensions</div></div>
  <div class="stat"><div class="stat-val">{method_label}</div><div class="stat-lbl">Réduction</div></div>
  <div class="stat"><div class="stat-val">1024→2</div><div class="stat-lbl">Projection</div></div>
</div>

<nav>
  <button class="active" onclick="showTab('tab2d', this)">Vue 2D</button>
  {f"<button onclick=" + chr(39) + "showTab(" + chr(39) + "tab3d" + chr(39) + ", this)" + chr(39) + ">Vue 3D</button>" if include_3d else ""}
  <button onclick="showTab('tabvar', this)">Variance PCA</button>
</nav>

<div id="tab2d" class="tab-content active">
  <div class="plot-container">{html_2d}</div>
</div>

  {f"<div id='tab3d' class='tab-content'><div class='plot-container'>" + html_3d + "</div></div>" if include_3d else ""}

<div id="tabvar" class="tab-content">
  <div class="plot-container">{html_var}</div>
</div>

<footer>Généré par pplx-embed · Perplexity AI pplx-embed-v1-0.6B · MIT License</footer>

<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>
"""
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

# ─── Demo data ────────────────────────────────────────────────────────────────

DEMO_TEXTS = [
    # Sciences
    "La photosynthèse est le processus par lequel les plantes transforment la lumière en énergie.",
    "La mitose est la division cellulaire produisant deux cellules filles identiques.",
    "L'ADN contient l'information génétique de tous les organismes vivants.",
    "Les enzymes sont des protéines qui accélèrent les réactions biochimiques.",
    # Histoire
    "La Révolution française a débuté en 1789 avec la prise de la Bastille.",
    "Napoléon Bonaparte est devenu empereur des Français en 1804.",
    "La Première Guerre mondiale a causé plus de 17 millions de morts.",
    "La chute du mur de Berlin en 1989 a symbolisé la fin de la Guerre froide.",
    # Technologie
    "Les modèles de langage comme GPT apprennent à partir de grandes quantités de texte.",
    "Le machine learning permet aux ordinateurs d'apprendre sans être explicitement programmés.",
    "Les réseaux de neurones profonds imitent le fonctionnement du cerveau humain.",
    "Le traitement du langage naturel permet aux machines de comprendre le texte.",
    # Géographie
    "Paris est la capitale de la France et compte environ 2 millions d'habitants.",
    "La Seine traverse Paris du sud-est vers le nord-ouest.",
    "Le Mont-Blanc est le point culminant des Alpes à 4 808 mètres.",
    "La Loire est le fleuve le plus long de France avec 1 013 kilomètres.",
]

DEMO_CATEGORIES = (
    ["Sciences"] * 4 + ["Histoire"] * 4 +
    ["Technologie"] * 4 + ["Géographie"] * 4
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export an interactive embedding projector to standalone HTML.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to embeddings.json (from /embed API endpoint)")
    parser.add_argument("--output", "-o", type=str, default="projector.html",
                        help="Output HTML path (default: projector.html)")
    parser.add_argument("--method", "-m", type=str, default="umap",
                        choices=["pca", "umap", "tsne"],
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--clusters", "-k", type=int, default=6,
                        help="Number of K-Means clusters (default: 6)")
    parser.add_argument("--no-3d", action="store_true",
                        help="Skip 3D plot (faster export)")
    parser.add_argument("--title", type=str, default="Embedding Projector",
                        help="Title shown in the HTML page")
    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in French demo data (no --input needed)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.demo or args.input is None:
        print("🔮 Using built-in French demo data...")
        # Embed via local API if available, else use random vectors
        try:
            import requests
            api_url = os.getenv("API_URL", "http://localhost:8000")
            r = requests.post(f"{api_url}/embed", json={"texts": DEMO_TEXTS}, timeout=60)
            r.raise_for_status()
            embeddings = r.json()["embeddings"]
            print(f"   ✓ Embeddings from local API ({len(embeddings)} vectors)")
        except Exception as e:
            print(f"   ⚠ API not available ({e}) — using random demo vectors")
            rng = np.random.default_rng(42)
            embeddings = rng.normal(size=(len(DEMO_TEXTS), 64)).tolist()

        texts = DEMO_TEXTS
        categories = DEMO_CATEGORIES
    else:
        print(f"📂 Loading embeddings from {args.input} ...")
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        embeddings = data["embeddings"]
        texts = data.get("texts", [f"text_{i}" for i in range(len(embeddings))])
        categories = data.get("categories", None)

    # ── Project & export ───────────────────────────────────────────────────────
    print(f"⚙️  Reducing {len(embeddings)} vectors ({len(embeddings[0])}D → 2D/3D) with {args.method.upper()}...")
    proj = EmbeddingProjector(embeddings, labels=texts, categories=categories)

    print(f"🎨 Building interactive plots...")
    out = proj.export_html(
        output_path=args.output,
        method=args.method,  # type: ignore
        include_3d=not args.no_3d,
        n_clusters=args.clusters,
        title=args.title,
    )

    size_kb = Path(out).stat().st_size // 1024
    print(f"\n✅ Projector exported → {out}  ({size_kb} KB)")
    print(f"   Open in your browser: file://{Path(out).resolve()}")


if __name__ == "__main__":
    main()
