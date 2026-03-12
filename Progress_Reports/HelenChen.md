# Network-Based Correlation Analysis

**Script:** `analysis/Network Analysis/network_correlation.py`
**Outputs:** `analysis/Network Analysis/outputs/`

Extends the EDA correlation heatmap into a **graph-based representation** of feature relationships, directly addressing the project guiding principle of network-based analysis. The network visually shows which feature clusters sit closest to the Bull/Bear regime label, complementing the Feature Importance and Granger Causality analyses.

---

## Methodology

- Built an **undirected weighted correlation graph** from 312 monthly observations (Jan 2000 – Jan 2026) using 11 features + the Regime label as nodes.
- **Edge inclusion threshold**: |r| ≥ 0.30 — only meaningful correlations are drawn.
- **Edge width** ∝ |correlation| strength; **edge colour** = blue (positive) / red (negative).
- **Node colour** = feature category (blue = technical, orange = macro, red = Fed, green = Regime).
- Layout: **force-directed spring layout** — tightly correlated features naturally cluster together without manual positioning.
- Produced two plots:
  1. **Full network** — all 12 nodes, 35 edges, with strong correlations (|r| ≥ 0.60) labelled.
  2. **Ego network** — Regime node and its 5 direct neighbours only, with a ranked bar chart of correlations on the right.

---

## Results

### Network Structure

| Metric | Value |
|---|---|
| Nodes | 12 |
| Edges at \|r\| ≥ 0.30 | 35 |
| Most connected node | SPX_MACD (degree = 9) |
| Least connected node | Fed_Funds_Rate (degree = 3) |

### Feature Correlations with Regime Label (ranked by |r|)

| Feature | r | Category |
|---|---|---|
| SPX_ROC | +0.680 | Technical |
| SPX_RSI | +0.664 | Technical |
| SPX_MACDH | +0.602 | Technical |
| VIX_Close | −0.520 | Technical |
| SPX_MACD | +0.372 | Technical |
| Real_GDP | +0.209 | Macro |
| Inflation | +0.161 | Macro |
| SPX_Volume | +0.077 | Technical |
| 10Y2Y_Spread | −0.050 | Macro |
| Unemployment | −0.031 | Macro |
| Fed_Funds_Rate | −0.022 | Fed |

### Average |correlation| with Regime by category

| Category | Avg \|r\| |
|---|---|
| Technical | 0.486 |
| Macro | 0.113 |
| Fed | 0.022 |

---

## Key Findings

1. **Technical indicators form a tight cluster around the Regime node.** SPX_ROC, SPX_RSI, SPX_MACDH, and VIX all have |r| > 0.50 with the Regime label. In the spring layout, these nodes are visually pulled close to the Regime node, making the dominance of momentum signals immediately apparent.

2. **Fed_Funds_Rate is the most isolated node in the graph.** With only 3 edges and a near-zero correlation with Regime (r = −0.022), it forms no meaningful cluster with either technical or macro variables. This is the network-level equivalent of what L1 pruning and feature importance rankings found.

3. **Macro variables form their own loose cluster, separate from Regime.** Real_GDP and Inflation show moderate inter-correlations with each other, but neither connects strongly to the Regime node. The macro cluster sits at a distance from the regime label in the spring layout.

4. **VIX is the only technical indicator with a negative correlation with Regime (r = −0.520).** This makes economic sense — high volatility is associated with bear markets (Regime = 0). Its negative edge (red) is visually distinct from the positive momentum edges (blue), reflecting that fear and momentum are complementary signals.

---

## Interpretation

The network graph provides a **visual confirmation** of findings from Feature Importance and Granger Causality:

> *Market momentum and volatility signals (RSI, ROC, MACDH, VIX) cluster tightly around the regime label. Macro indicators — including the Fed Funds Rate — form a separate, loosely connected subgraph with little direct link to regime states. The network makes this structural separation immediately interpretable to a non-technical audience.*

The ego network plot is particularly suitable for the presentation slide, as it isolates exactly which features connect to the regime node and ranks them in a single combined figure.

---

## Limitations

- **Pearson correlation only** — the network captures linear relationships. Non-linear dependencies (e.g. regime-switching dynamics during crises) are not reflected.
- **Static correlation** — computed over the full 2000–2026 period. The correlation structure may differ across sub-periods (e.g. pre/post-2008, pre/post-COVID). A rolling network analysis would capture this but was not implemented here.
- **No significance testing on edges** — edges are included based on a |r| ≥ 0.30 threshold rather than a formal p-value correction. With 66 possible pairs, some edges may be spurious. A Bonferroni or FDR-corrected threshold could be applied in future.
