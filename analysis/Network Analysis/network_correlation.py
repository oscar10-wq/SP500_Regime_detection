"""
Network-Based Correlation Analysis for S&P 500 Regime Detection
================================================================
Builds a correlation network where:
  - Nodes  = features + regime label
  - Edges  = pairs with |correlation| above a threshold
  - Layout = force-directed (spring), so tightly correlated
             features naturally cluster together

Research question addressed:
  Do technical indicators or macro indicators cluster more closely
  with the Bull/Bear regime label?

Run from the project root:
    python "analysis/Network Analysis/network_correlation.py"
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.scripts import data_generation
from data.scripts.analysis_utils import classify_regimes

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
START_DATE       = "2000-01-01"
END_DATE         = "2026-01-01"
EDGE_THRESHOLD   = 0.30   # minimum |correlation| to draw an edge
STRONG_THRESHOLD = 0.60   # edges above this are drawn thicker + labelled

# ── Feature categories (for node colouring) ───────────────────────────────────
CATEGORY = {
    "SPX_Volume":     "technical",
    "SPX_ROC":        "technical",
    "SPX_RSI":        "technical",
    "SPX_MACD":       "technical",
    "SPX_MACDH":      "technical",
    "VIX_Close":      "technical",
    "Real_GDP":       "macro",
    "Unemployment":   "macro",
    "Inflation":      "macro",
    "10Y2Y_Spread":   "macro",
    "Fed_Funds_Rate": "fed",
    "Regime":         "regime",
}

COLORS = {
    "technical": "#3498DB",   # blue
    "macro":     "#E67E22",   # orange
    "fed":       "#E74C3C",   # red
    "regime":    "#2ECC71",   # green — the target node
}

# Dark background style (consistent with other scripts)
BG   = "#0D1117"
CARD = "#161B22"
TEXT = "#C9D1D9"


# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════
def _build_dataset() -> pd.DataFrame:
    technical = data_generation.get_yahoo_finance_data(START_DATE, END_DATE)
    macro     = data_generation.get_fred_input_data(START_DATE, END_DATE)
    df        = pd.concat([technical, macro], axis=1)

    df["Regime"] = classify_regimes(df).astype(float)

    # Keep only the features we categorise (drops SPX_Close, SPX_MACDS, futures)
    cols = [c for c in CATEGORY if c in df.columns]
    df   = df[cols].dropna()
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ═════════════════════════════════════════════════════════════════════════════
def build_graph(corr: pd.DataFrame, threshold: float) -> nx.Graph:
    """
    Create an undirected weighted graph from a correlation matrix.
    Only edges with |corr| >= threshold are added.
    Edge weight  = absolute correlation value (used for layout force).
    Edge sign    = stored separately (positive / negative).
    """
    G = nx.Graph()
    G.add_nodes_from(corr.columns)

    for i, feat_a in enumerate(corr.columns):
        for feat_b in corr.columns[i + 1:]:
            r = corr.loc[feat_a, feat_b]
            if abs(r) >= threshold:
                G.add_edge(feat_a, feat_b, weight=abs(r), sign=np.sign(r), raw=r)
    return G


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Full correlation network
# ═════════════════════════════════════════════════════════════════════════════
def plot_full_network(G: nx.Graph, corr: pd.DataFrame, save_path: str):
    """
    Spring-layout network coloured by feature category.
    Edge width  ∝ |correlation|.
    Edge colour = blue (positive) / red (negative).
    """
    fig, ax = plt.subplots(figsize=(14, 11), facecolor=BG)
    ax.set_facecolor(BG)
    fig.suptitle(
        f"Feature Correlation Network  (|r| ≥ {EDGE_THRESHOLD})\n"
        "Node colour = feature category  |  Edge width ∝ |correlation|  |  "
        "Blue edge = positive  Red edge = negative",
        color=TEXT, fontsize=11, fontweight="bold", y=0.99,
    )

    # Position: weight edges so correlated features pull together
    pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5)

    node_colors = [COLORS.get(CATEGORY.get(n, "technical"), "#888") for n in G.nodes()]
    node_sizes  = [900 if n == "Regime" else 500 for n in G.nodes()]

    # Split edges by sign for colouring
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
    pos_widths = [G[u][v]["weight"] * 6 for u, v in pos_edges]
    neg_widths = [G[u][v]["weight"] * 6 for u, v in neg_edges]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_widths,
                           edge_color="#3498DB", alpha=0.55, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_widths,
                           edge_color="#E74C3C", alpha=0.55, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           ax=ax, edgecolors="#30363D", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color=TEXT,
                            font_weight="bold", ax=ax)

    # Edge weight labels for strong correlations only
    strong_labels = {
        (u, v): f"{d['raw']:.2f}"
        for u, v, d in G.edges(data=True)
        if abs(d["raw"]) >= STRONG_THRESHOLD
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=strong_labels,
                                 font_size=6.5, font_color=TEXT, ax=ax)

    # Legend
    handles = [mpatches.Patch(color=COLORS[cat], label=cat.capitalize())
               for cat in ["technical", "macro", "fed", "regime"]]
    handles += [
        mpatches.Patch(color="#3498DB", label="Positive correlation"),
        mpatches.Patch(color="#E74C3C", label="Negative correlation"),
    ]
    ax.legend(handles=handles, loc="lower left", facecolor=CARD,
              labelcolor=TEXT, fontsize=8, framealpha=0.9)

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Regime-centred ego network
# ═════════════════════════════════════════════════════════════════════════════
def plot_ego_network(G: nx.Graph, corr: pd.DataFrame, save_path: str):
    """
    Show only the Regime node and its direct neighbours.
    Bars on the right show correlation magnitude by feature category.
    This directly answers: which features drive regime classification?
    """
    if "Regime" not in G.nodes():
        print("       Regime node not in graph — skipping ego network.")
        return

    ego   = nx.ego_graph(G, "Regime")
    neighbors = [n for n in ego.nodes() if n != "Regime"]

    # Sort neighbours by absolute correlation with Regime
    reg_corr = corr["Regime"].drop("Regime").reindex(neighbors).sort_values(key=abs, ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG,
                             gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(
        "Regime Node — Ego Network & Correlation Ranking\n"
        "Left: network of direct neighbours  |  Right: correlation with Regime label",
        color=TEXT, fontsize=11, fontweight="bold", y=1.01,
    )

    # ── Left panel: ego network ───────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)

    pos = nx.spring_layout(ego, weight="weight", seed=42, k=3.0)
    node_colors = [
        COLORS["regime"] if n == "Regime"
        else COLORS.get(CATEGORY.get(n, "technical"), "#888")
        for n in ego.nodes()
    ]
    node_sizes = [1200 if n == "Regime" else 600 for n in ego.nodes()]

    pos_edges = [(u, v) for u, v, d in ego.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in ego.edges(data=True) if d["sign"] < 0]

    nx.draw_networkx_edges(ego, pos, edgelist=pos_edges,
                           width=[ego[u][v]["weight"] * 7 for u, v in pos_edges],
                           edge_color="#3498DB", alpha=0.6, ax=ax)
    nx.draw_networkx_edges(ego, pos, edgelist=neg_edges,
                           width=[ego[u][v]["weight"] * 7 for u, v in neg_edges],
                           edge_color="#E74C3C", alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(ego, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax,
                           edgecolors="#30363D", linewidths=1.2)
    nx.draw_networkx_labels(ego, pos, font_size=8.5, font_color=TEXT,
                            font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"{d['raw']:.2f}" for u, v, d in ego.edges(data=True)}
    nx.draw_networkx_edge_labels(ego, pos, edge_labels=edge_labels,
                                 font_size=7, font_color=TEXT, ax=ax)
    ax.axis("off")
    handles = [mpatches.Patch(color=COLORS[cat], label=cat.capitalize())
               for cat in ["technical", "macro", "fed", "regime"]]
    ax.legend(handles=handles, loc="lower left", facecolor=CARD,
              labelcolor=TEXT, fontsize=7.5, framealpha=0.9)

    # ── Right panel: ranked bar chart ────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(CARD)

    bar_colors = [
        COLORS["regime"] if f == "Regime"
        else COLORS.get(CATEGORY.get(f, "technical"), "#888")
        for f in reg_corr.index
    ]
    bars = ax2.barh(reg_corr.index, reg_corr.values,
                    color=bar_colors, edgecolor="#30363D", height=0.65)

    # Value labels on bars
    for bar, val in zip(bars, reg_corr.values):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha    = "left"      if val >= 0 else "right"
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", ha=ha, color=TEXT, fontsize=8)

    ax2.axvline(0, color=TEXT, lw=0.8, alpha=0.5)
    ax2.set_xlabel("Pearson Correlation with Regime", color=TEXT)
    ax2.set_title("Feature Correlation with Regime Label\n(sorted by |r|)",
                  color=TEXT, fontsize=9, fontweight="bold")
    ax2.tick_params(colors=TEXT, labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color("#30363D")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def _print_summary(G: nx.Graph, corr: pd.DataFrame):
    w = 72
    print(f"\n{'─'*w}")
    print(f"  NETWORK SUMMARY")
    print(f"{'─'*w}")
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}  (|r| ≥ {EDGE_THRESHOLD})")

    # Degree centrality — most connected nodes
    degree = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Node degree (most connected first):")
    for node, deg in degree:
        cat = CATEGORY.get(node, "?")
        print(f"    {node:<20} degree={deg}  [{cat}]")

    # Regime correlations
    if "Regime" in corr.columns:
        reg_corr = (corr["Regime"]
                    .drop("Regime")
                    .sort_values(key=abs, ascending=False))
        print(f"\n  Correlations with Regime label (all features, ranked by |r|):")
        for feat, r in reg_corr.items():
            cat  = CATEGORY.get(feat, "?")
            flag = " ◀" if abs(r) >= EDGE_THRESHOLD else ""
            print(f"    {feat:<20} r={r:+.3f}  [{cat}]{flag}")

    # Category-level summary: avg |corr| with Regime
    print(f"\n  Average |correlation| with Regime by category:")
    if "Regime" in corr.columns:
        for cat in ["technical", "macro", "fed"]:
            feats   = [f for f, c in CATEGORY.items() if c == cat and f in corr.columns]
            avg_abs = corr.loc[feats, "Regime"].abs().mean()
            print(f"    {cat:<12} avg |r| = {avg_abs:.3f}")

    print(f"\n{'─'*w}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    w = 72
    print(f"\n{'═'*w}\n  NETWORK-BASED CORRELATION ANALYSIS\n{'═'*w}")

    print("\n[1/4]  Loading data...")
    df = _build_dataset()
    print(f"       {len(df)} monthly observations  |  {len(df.columns)} features + Regime")

    print("\n[2/4]  Computing correlation matrix...")
    corr = df.corr()

    print("\n[3/4]  Building graph...")
    G = build_graph(corr, threshold=EDGE_THRESHOLD)
    print(f"       {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"at threshold |r| ≥ {EDGE_THRESHOLD}")

    _print_summary(G, corr)

    print("\n[4/4]  Generating plots...")
    plot_full_network(G, corr,
                      os.path.join(OUTPUT_DIR, "1_full_network.png"))
    plot_ego_network(G, corr,
                     os.path.join(OUTPUT_DIR, "2_regime_ego_network.png"))

    print(f"\n{'═'*w}")
    print(f"  Done. Plots saved to: {OUTPUT_DIR}/")
    print(f"{'═'*w}\n")

    return dict(df=df, corr=corr, G=G)


if __name__ == "__main__":
    run()
