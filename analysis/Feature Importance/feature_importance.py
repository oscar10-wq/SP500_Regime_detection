"""
Feature Importance Analysis for S&P 500 Regime Detection
=========================================================
Investigates whether Federal Reserve interest rate hikes are
the main trigger for bull-to-bear regime switches.

Run this script from the project root:
    python feature_importance.py

Or copy the sections into project.ipynb under "Section IV: Feature Importance".
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from data.scripts import data_generation

# ── Global Styling ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 130,
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

PALETTE = {
    "fed":    "#e74c3c",   # red — highlight for Fed Funds Rate
    "macro":  "#e67e22",   # orange — other macro indicators
    "tech":   "#3498db",   # blue — technical indicators
    "bull":   "#27ae60",   # green
    "bear":   "#c0392b",   # dark red
    "pos":    "#2ecc71",   # positive coefficient
    "neg":    "#e74c3c",   # negative coefficient
}


# ════════════════════════════════════════════════════════════════════════
#  1) LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════════════════

print("Loading data...")
start_date, end_date = "2000-01-01", "2026-01-01"

technical = data_generation.get_yahoo_finance_data(start_date, end_date)
macro     = data_generation.get_fred_input_data(start_date, end_date)

full_data = pd.concat([technical, macro], axis=1)
full_data["Regime"] = data_generation.classify_regimes(full_data)
full_data.dropna(inplace=True)

# We exclude SPX_Close from the feature set because the regime labels
# are derived directly from it (20% peak-to-trough rule). Including it
# would cause data leakage.
FEATURE_COLS = [
    "SPX_Volume", "SPX_ROC", "SPX_RSI",
    "SPX_MACD", "SPX_MACDH", "SPX_MACDS",
    "VIX_Close",
    "Real_GDP", "Unemployment", "Inflation",
    "Fed_Funds_Rate", "10Y2Y_Spread",
]

# Categorise features for colour-coding
MACRO_FEATURES = {"Real_GDP", "Unemployment", "Inflation", "Fed_Funds_Rate", "10Y2Y_Spread"}

X = full_data[FEATURE_COLS]
y = full_data["Regime"]

# Chronological train/test split (80/20) — no future data leakage
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=FEATURE_COLS, index=X_test.index)

print(f"Train: {len(X_train)} months  |  Test: {len(X_test)} months")
print(f"Train regime balance — Bull: {int((y_train==1).sum())}  Bear: {int((y_train==0).sum())}")
print(f"Test  regime balance — Bull: {int((y_test==1).sum())}   Bear: {int((y_test==0).sum())}")


# ════════════════════════════════════════════════════════════════════════
#  2) TRAIN THREE CLASSIFIERS
# ════════════════════════════════════════════════════════════════════════

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"\n{'─'*50}")
    print(f"  {name}  —  Accuracy: {acc:.2%}")
    print(f"{'─'*50}")
    print(classification_report(y_test, preds, target_names=["Bear", "Bull"], zero_division=0))


# ════════════════════════════════════════════════════════════════════════
#  3) FEATURE IMPORTANCE — RANDOM FOREST
# ════════════════════════════════════════════════════════════════════════

rf = results["Random Forest"]["model"]
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()

def feature_color(name):
    if name == "Fed_Funds_Rate":
        return PALETTE["fed"]
    elif name in MACRO_FEATURES:
        return PALETTE["macro"]
    return PALETTE["tech"]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(imp.index, imp.values, color=[feature_color(f) for f in imp.index], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Importance (Mean Decrease in Impurity)")
ax.set_title("Random Forest — Feature Importance for Regime Detection")

# Legend
legend_handles = [
    mpatches.Patch(color=PALETTE["fed"],   label="Fed Funds Rate"),
    mpatches.Patch(color=PALETTE["macro"], label="Other Macro Indicators"),
    mpatches.Patch(color=PALETTE["tech"],  label="Technical Indicators"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n📊  Random Forest — Feature Importance Ranking:")
print("=" * 48)
for rank, (feat, score) in enumerate(imp.sort_values(ascending=False).items(), 1):
    tag = " ◀ FED RATE" if feat == "Fed_Funds_Rate" else ""
    print(f"  {rank:>2}. {feat:<20s} {score:.4f}{tag}")


# ════════════════════════════════════════════════════════════════════════
#  4) FEATURE IMPORTANCE — GRADIENT BOOSTING
# ════════════════════════════════════════════════════════════════════════

gb = results["Gradient Boosting"]["model"]
gb_imp = pd.Series(gb.feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(gb_imp.index, gb_imp.values, color=[feature_color(f) for f in gb_imp.index], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Importance (Mean Decrease in Impurity)")
ax.set_title("Gradient Boosting — Feature Importance for Regime Detection")
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("gb_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()


# ════════════════════════════════════════════════════════════════════════
#  5) LOGISTIC REGRESSION COEFFICIENTS
# ════════════════════════════════════════════════════════════════════════

lr = results["Logistic Regression"]["model"]
coefs = pd.Series(lr.coef_[0], index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
colors = []
for feat, val in zip(coefs.index, coefs.values):
    if feat == "Fed_Funds_Rate":
        colors.append(PALETTE["fed"])
    elif val > 0:
        colors.append(PALETTE["pos"])
    else:
        colors.append(PALETTE["neg"])

ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Coefficient Value (on standardised features)")
ax.set_title("Logistic Regression — Coefficient Direction & Magnitude")

coef_legend = [
    mpatches.Patch(color=PALETTE["fed"], label="Fed Funds Rate"),
    mpatches.Patch(color=PALETTE["pos"], label="Positive (→ Bull)"),
    mpatches.Patch(color=PALETTE["neg"], label="Negative (→ Bear)"),
]
ax.legend(handles=coef_legend, loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("lr_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n📊  Logistic Regression — Coefficient Ranking (absolute value):")
print("=" * 55)
abs_coefs = coefs.abs().sort_values(ascending=False)
for rank, (feat, val) in enumerate(abs_coefs.items(), 1):
    direction = "→ Bull" if coefs[feat] > 0 else "→ Bear"
    tag = " ◀ FED RATE" if feat == "Fed_Funds_Rate" else ""
    print(f"  {rank:>2}. {feat:<20s} |coef| = {val:.4f}  ({direction}){tag}")


# ════════════════════════════════════════════════════════════════════════
#  6) CROSS-MODEL CONSENSUS — WHERE DOES FED FUNDS RATE RANK?
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  CROSS-MODEL CONSENSUS: Fed Funds Rate Ranking")
print("=" * 60)

rf_rank  = (imp.sort_values(ascending=False).index.tolist().index("Fed_Funds_Rate")) + 1
gb_rank  = (gb_imp.sort_values(ascending=False).index.tolist().index("Fed_Funds_Rate")) + 1
lr_rank  = (abs_coefs.index.tolist().index("Fed_Funds_Rate")) + 1

print(f"  Logistic Regression :  #{lr_rank} / {len(FEATURE_COLS)}")
print(f"  Random Forest       :  #{rf_rank} / {len(FEATURE_COLS)}")
print(f"  Gradient Boosting   :  #{gb_rank} / {len(FEATURE_COLS)}")
print(f"\n  Average Rank: {(rf_rank + gb_rank + lr_rank) / 3:.1f} / {len(FEATURE_COLS)}")
print("=" * 60)

if (rf_rank + gb_rank + lr_rank) / 3 <= 3:
    print("\n  ✅ FINDING: Fed Funds Rate is among the TOP drivers of regime switches.")
elif (rf_rank + gb_rank + lr_rank) / 3 <= 6:
    print("\n  ⚠️  FINDING: Fed Funds Rate is a MODERATE driver — not the dominant one.")
else:
    print("\n  ❌ FINDING: Fed Funds Rate is a WEAK driver of regime switches.")


# ════════════════════════════════════════════════════════════════════════
#  7) CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["preds"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Bear", "Bull"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.2%}", fontsize=12, fontweight="bold")
plt.suptitle("Confusion Matrices — Test Set Performance", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ Analysis complete. Charts saved as .png files in the project root.")
