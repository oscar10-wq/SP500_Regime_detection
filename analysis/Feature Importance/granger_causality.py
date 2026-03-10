

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from sklearn.metrics import confusion_matrix
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.scripts import data_generation
from data.scripts.analysis_utils import classify_regimes

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
LAGS          = [1, 2, 3, 6, 12]
START_DATE    = "2000-01-01"
END_DATE      = "2026-01-01"
TRAIN_CUTOFF  = "2013-01-01"      # everything before this = train
# 2013 cutoff gives ~144 train obs (enough for 35-param Fed HMM at ~4× rule of thumb)
# and ~156 test obs (vs 120 with 2016) — meaningfully more power for OOS Granger at lag 12.
HMM_N_ITER    = 1000
HMM_N_RANDOM  = 10               # random restarts per model

CONTROL_COLS  = ["SPX_Return", "VIX_Close", "SPX_RSI"]
CAUSE_COLS    = ["Fed_Cycle", "Fed_Change", "Fed_Funds_Rate", "10Y2Y_Spread"]

# ── Colours ───────────────────────────────────────────────────────────────────
BG   = "#0D1117"; CARD = "#161B22"; TEXT = "#C9D1D9"
BULL = "#2ECC71"; BEAR = "#E74C3C"; FED  = "#3498DB"
SPRD = "#F39C12"; NEUT = "#95A5A6"
CAUSE_COLORS = {
    "Fed_Cycle":      "#9B59B6",
    "Fed_Change":     "#E67E22",
    "Fed_Funds_Rate": FED,
    "10Y2Y_Spread":   SPRD,
}


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def _build_dataset() -> pd.DataFrame:
    technical = data_generation.get_yahoo_finance_data(START_DATE, END_DATE)
    macro     = data_generation.get_fred_input_data(START_DATE, END_DATE)
    df        = pd.concat([technical, macro], axis=1)

    if "SPX_Close" not in df.columns:
        raise ValueError("SPX_Close is required.")

    # classify_regimes returns 1=Bull, 0=Bear → flip to 0=Bull, 1=Bear
    df["bear_rule"] = (1 - classify_regimes(df)).astype(int)
    df["SPX_Return"] = np.log(df["SPX_Close"]).diff()
    df["SPX_RV_20"]  = df["SPX_Return"].rolling(20).std()

    if "VIX_Close" in df.columns:
        df["VIX_Change"] = df["VIX_Close"].diff()
    if "Fed_Funds_Rate" in df.columns:
        df["Fed_Change"] = df["Fed_Funds_Rate"].diff()
        df["Fed_Cycle"]  = _compute_fed_cycle(df["Fed_Change"])

    return df


def _compute_fed_cycle(fed_change: pd.Series, threshold: float = 0.05) -> pd.Series:
    """
    Convert sparse Fed_Change into a persistent cycle direction.
    Fed_Change is 0 for ~60-70% of months; this carries the last-move
    direction forward: +1 = hiking cycle, -1 = cutting cycle.
    """
    cycle    = pd.Series(0.0, index=fed_change.index)
    last_dir = 0.0
    for i, ch in enumerate(fed_change.values):
        if ch > threshold:
            last_dir = 1.0
        elif ch < -threshold:
            last_dir = -1.0
        cycle.iloc[i] = last_dir
    return cycle


full = _build_dataset()


# ═════════════════════════════════════════════════════════════════════════════
#  SCALING  (fit on train, apply to train/test/full — no leakage)
# ═════════════════════════════════════════════════════════════════════════════
def zscore_fit(df: pd.DataFrame, cols: list):
    X = df[cols].astype(float).values
    return X.mean(0), X.std(0) + 1e-8


def zscore_apply(df: pd.DataFrame, cols: list, mu, sd) -> np.ndarray:
    return (df[cols].astype(float).values - mu) / sd


# ═════════════════════════════════════════════════════════════════════════════
#  HMM HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def fit_hmm(X: np.ndarray, name: str, init_labels: np.ndarray = None) -> GaussianHMM:
    best_ll = -np.inf
    best_m  = None

    # ── label-seeded start ───────────────────────────────────────────────────
    if init_labels is not None:
        m = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=HMM_N_ITER, random_state=0,
                        init_params="stc")       # skip mean init ("m")
        m.means_ = np.array([
            X[init_labels == 0].mean(axis=0),   # Bull centre
            X[init_labels == 1].mean(axis=0),   # Bear centre
        ])
        m.fit(X)
        ll = m.score(X) * len(X)
        if ll > best_ll:
            best_ll, best_m = ll, m

    # ── random restarts ──────────────────────────────────────────────────────
    for seed in range(HMM_N_RANDOM):
        m = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=HMM_N_ITER, random_state=seed + 1)
        m.fit(X)
        ll = m.score(X) * len(X)
        if ll > best_ll:
            best_ll, best_m = ll, m

    best_m.name = name
    return best_m


def align_bear_bull(m: GaussianHMM, vix_idx: int) -> GaussianHMM:
    """Ensure state 0 = Bull (low VIX mean), state 1 = Bear (high VIX mean)."""
    if m.means_[0][vix_idx] > m.means_[1][vix_idx]:
        m.means_     = m.means_[[1, 0]].copy()
        # Use _covars_ directly — hmmlearn's covars_ setter has a validation
        # quirk in 0.3.x that rejects the swap when covariance_type="diag".
        m._covars_   = m._covars_[[1, 0]].copy()
        m.startprob_ = m.startprob_[[1, 0]].copy()
        m.transmat_  = m.transmat_[[1, 0]][:, [1, 0]].copy()
    return m


def hmm_info_criteria(m: GaussianHMM, X: np.ndarray):
    """
    Return (log-likelihood, AIC, BIC, converged).
    Handles both 'full' and 'diag' covariance types for parameter count.
    Can be called with training data (in-sample) or test data (OOS LL).
    """
    T, D = X.shape
    K    = m.n_components
    n_cov = K * D if m.covariance_type == "diag" else K * D * (D + 1) // 2
    n_par = (K - 1) + K * (K - 1) + K * D + n_cov
    ll    = m.score(X)       # score() returns total log-likelihood already
    aic   = -2 * ll + 2 * n_par
    bic   = -2 * ll + n_par * np.log(T)
    return ll, aic, bic, m.monitor_.converged


# ═════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL GRANGER CAUSALITY + BH CORRECTION
# ═════════════════════════════════════════════════════════════════════════════
def _ols(A, y):
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    rss = res @ res
    sst = ((y - y.mean()) ** 2).sum()
    return rss, (1 - rss / sst) if sst > 0 else 0.0


def conditional_granger_test(y, x_cause, x_controls, max_lag):
    """
    Conditional Granger F-test.

    Restricted:   Y ~ const + Y[t-1..p] + controls[t-1..p]
    Unrestricted: Y ~ const + Y[t-1..p] + controls[t-1..p] + X_cause[t-1..p]

    F = ((RSS_R − RSS_U) / p) / (RSS_U / df2)
    """
    rows, n, n_ctrl = [], len(y), len(x_controls)

    for p in range(1, max_lag + 1):
        T        = n - p
        n_params = 1 + p + p * n_ctrl + p
        if T <= n_params + 5:
            continue

        Y    = y[p:]
        ones = np.ones((T, 1))
        Yr   = np.column_stack([y[p - k - 1: n - k - 1] for k in range(p)])
        Cr   = np.column_stack([
            ctrl[p - k - 1: n - k - 1]
            for ctrl in x_controls for k in range(p)
        ])
        Xr   = np.column_stack([x_cause[p - k - 1: n - k - 1] for k in range(p)])

        base        = np.hstack([ones, Yr, Cr])
        rss_r, r2_r = _ols(base, Y)
        rss_u, r2_u = _ols(np.hstack([base, Xr]), Y)

        df1 = p
        df2 = T - n_params
        if df2 <= 0:
            continue

        F  = ((rss_r - rss_u) / df1) / (rss_u / df2)
        pv = 1 - stats.f.cdf(F, df1, df2)
        rows.append(dict(lag=p, F=F, p=pv, p_adj=pv,
                         r2_restricted=r2_r, r2_unrestricted=r2_u,
                         r2_gain=r2_u - r2_r))
    return rows


def bh_correct(all_gc: dict) -> dict:
    """
    Benjamini-Hochberg FDR correction across ALL tests in all_gc.
    Updates each row's 'p_adj' field in-place on a copy.
    """
    # Gather (p, cause, tgt, row_idx)
    entries = [
        (r["p"], cause, tgt, i)
        for (cause, tgt), rows in all_gc.items()
        for i, r in enumerate(rows)
    ]
    if not entries:
        return all_gc

    m     = len(entries)
    order = sorted(range(m), key=lambda i: entries[i][0])

    p_adj = [0.0] * m
    for rank, i in enumerate(order, start=1):
        p_adj[i] = min(1.0, entries[i][0] * m / rank)

    # Enforce monotonicity (adjusted p cannot decrease as raw p increases)
    for k in range(len(order) - 2, -1, -1):
        p_adj[order[k]] = min(p_adj[order[k]], p_adj[order[k + 1]])

    result = {k: [r.copy() for r in v] for k, v in all_gc.items()}
    for idx, (_, cause, tgt, row_i) in enumerate(entries):
        result[(cause, tgt)][row_i]["p_adj"] = p_adj[idx]

    return result


def _run_granger(df_seg: pd.DataFrame, rule_bear: np.ndarray,
                 correct: bool = True) -> dict:
    ctrl_arrays = [df_seg[c].values for c in CONTROL_COLS]
    y = rule_bear.astype(float)
    all_gc = {}
    for cause in CAUSE_COLS:
        x    = df_seg[cause].values
        rows = conditional_granger_test(y, x, ctrl_arrays, max(LAGS))
        all_gc[(cause, "Rule Regime")] = [r for r in rows if r["lag"] in LAGS]
    return bh_correct(all_gc) if correct else all_gc


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "ns"


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def run(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = df.copy().sort_index()

    MARKET_COLS = ["SPX_Return", "VIX_Close", "SPX_RSI", "SPX_ROC", "SPX_RV_20"]
    FED_COLS    = MARKET_COLS + ["Fed_Funds_Rate", "Fed_Change", "10Y2Y_Spread"]
    ALL_NEEDED  = MARKET_COLS + ["Fed_Funds_Rate", "Fed_Change", "Fed_Cycle",
                                  "10Y2Y_Spread", "bear_rule"]

    missing = [c for c in ALL_NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df.dropna(subset=ALL_NEEDED, inplace=True)

    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    tr = df[df.index <  cutoff].copy()
    te = df[df.index >= cutoff].copy()
    n_tr, n_te, T = len(tr), len(te), len(df)

    _header("FED RATE → BULL/BEAR REGIME ANALYSIS",
            f"Train: {tr.index[0].date()} → {tr.index[-1].date()}  ({n_tr} obs)  |  "
            f"Test: {te.index[0].date()} → {te.index[-1].date()}  ({n_te} obs)")

    # ── [1/5] Fit HMMs on TRAINING data ─────────────────────────────────────
    print(f"\n[1/5]  FITTING HMMs  (train only, {HMM_N_RANDOM} random restarts + label-seeded)")
    print(f"       Covariance type: diag  |  Market: {len(MARKET_COLS)} features  "
          f"|  Market+Fed: {len(FED_COLS)} features")

    mu_m, sd_m = zscore_fit(tr, MARKET_COLS)
    mu_f, sd_f = zscore_fit(tr, FED_COLS)

    Xm_tr  = zscore_apply(tr, MARKET_COLS, mu_m, sd_m)
    Xf_tr  = zscore_apply(tr, FED_COLS,    mu_f, sd_f)
    Xm_te  = zscore_apply(te, MARKET_COLS, mu_m, sd_m)
    Xf_te  = zscore_apply(te, FED_COLS,    mu_f, sd_f)
    Xm_all = zscore_apply(df, MARKET_COLS, mu_m, sd_m)

    rule_tr = tr["bear_rule"].values
    rule_te = te["bear_rule"].values
    rule_all = df["bear_rule"].values

    hmm_m = fit_hmm(Xm_tr, "Market-only", init_labels=rule_tr)
    hmm_m = align_bear_bull(hmm_m, MARKET_COLS.index("VIX_Close"))

    hmm_f = fit_hmm(Xf_tr, "Market+Fed",  init_labels=rule_tr)
    hmm_f = align_bear_bull(hmm_f, FED_COLS.index("VIX_Close"))

    # Information criteria: in-sample (train) and out-of-sample (test)
    ll_m_tr, aic_m_tr, bic_m_tr, conv_m = hmm_info_criteria(hmm_m, Xm_tr)
    ll_f_tr, aic_f_tr, bic_f_tr, conv_f = hmm_info_criteria(hmm_f, Xf_tr)
    ll_m_te, _, _, _                     = hmm_info_criteria(hmm_m, Xm_te)
    ll_f_te, _, _, _                     = hmm_info_criteria(hmm_f, Xf_te)

    print(f"\n       {'Model':<20} {'Train LL':>10}  {'AIC':>10}  {'BIC':>10}  "
          f"{'Test LL':>10}  {'LL/obs':>8}  {'Conv':>6}")
    print(f"       {'─' * 82}")
    for name, ll_tr, aic, bic, ll_te, n_te_obs, conv in [
        ("Market-only", ll_m_tr, aic_m_tr, bic_m_tr, ll_m_te, n_te, conv_m),
        ("Market+Fed",  ll_f_tr, aic_f_tr, bic_f_tr, ll_f_te, n_te, conv_f),
    ]:
        print(f"       {name:<20} {ll_tr:>10.2f}  {aic:>10.2f}  {bic:>10.2f}  "
              f"{ll_te:>10.2f}  {ll_te/n_te_obs:>8.3f}  {str(conv):>6}")

    dAIC_tr = aic_m_tr - aic_f_tr
    dBIC_tr = bic_m_tr - bic_f_tr
    dLL_te  = ll_f_te  - ll_m_te

    print(f"\n       In-sample  — ΔAIC={dAIC_tr:+.2f}, ΔBIC={dBIC_tr:+.2f}  "
          f"→ Fed {'IMPROVES fit ✓' if dAIC_tr > 0 else 'does NOT improve AIC'}")
    print(f"       Out-of-sample — ΔLL/obs(fed−market)={dLL_te/n_te:+.3f}  "
          f"→ Fed {'fits test data better ✓' if dLL_te > 0 else 'does NOT fit test data better'}")
    print(f"       Note: OOS LL difference reflects model complexity penalty, not Granger causality."
          f"\n              A simpler model (fewer params) almost always wins OOS LL with limited data.")

    # ── [2/5] Predict states ─────────────────────────────────────────────────
    print(f"\n[2/5]  REGIME COMPARISON  (0=Bull, 1=Bear)")

    # Full timeline (for plots)
    df["hmm_state"] = hmm_m.predict(Xm_all)
    df["bear_prob"]  = hmm_m.predict_proba(Xm_all)[:, 1]

    # Train period
    tr["hmm_state"] = hmm_m.predict(Xm_tr)
    tr["bear_prob"]  = hmm_m.predict_proba(Xm_tr)[:, 1]

    # Test period (OOS — the meaningful evaluation)
    te["hmm_state"] = hmm_m.predict(Xm_te)
    te["bear_prob"]  = hmm_m.predict_proba(Xm_te)[:, 1]

    def _regime_stats(label, seg, rule):
        n = len(seg)
        hmm_s = seg["hmm_state"].values
        agr   = (hmm_s == rule).mean()
        n_br  = rule.sum();   n_bh = hmm_s.sum()
        cm    = confusion_matrix(rule, hmm_s, labels=[0, 1])
        print(f"\n       {label}  (n={n})")
        print(f"         Rule-based:   {int(n_br):>4} bear / {n - int(n_br):>4} bull  ({n_br/n:.1%})")
        print(f"         HMM Viterbi:  {int(n_bh):>4} bear / {n - int(n_bh):>4} bull  ({n_bh/n:.1%})")
        print(f"         Agreement:    {agr:.1%}")
        print(f"                         HMM Bull   HMM Bear")
        print(f"           Rule Bull     {cm[0,0]:>6}     {cm[0,1]:>6}")
        print(f"           Rule Bear     {cm[1,0]:>6}     {cm[1,1]:>6}")
        return agr, cm

    agr_tr, cm_tr = _regime_stats("TRAIN period", tr, rule_tr)
    agr_te, cm_te = _regime_stats("TEST period  (out-of-sample)", te, rule_te)

    # ── [3/5] Granger causality ──────────────────────────────────────────────
    print(f"\n[3/5]  CONDITIONAL GRANGER CAUSALITY")
    print(f"       Controls: {', '.join(CONTROL_COLS)}")
    print(f"       Causes:   {', '.join(CAUSE_COLS)}")
    pct_zero = (df["Fed_Change"] == 0).mean()
    print(f"       Note: Fed_Change is 0 for {pct_zero:.1%} of months  "
          f"(Fed_Cycle is more persistent)")
    n_tests = len(CAUSE_COLS) * len(LAGS)
    print(f"       BH FDR correction across {n_tests} tests  (full sample only)")
    print(f"       Sub-periods: raw p-values, exploratory stability check")

    p1_label = f"{tr.index[0].year}–{tr.index[-1].year}  (n={n_tr})"
    p2_label = f"{te.index[0].year}–{te.index[-1].year}  (n={n_te})"

    gc_full = _run_granger(df,  rule_all, correct=True)   # BH corrected
    gc_p1   = _run_granger(tr,  rule_tr,  correct=False)  # raw p — stability check
    gc_p2   = _run_granger(te,  rule_te,  correct=False)  # raw p — stability check

    _print_granger_table(gc_full, f"Full sample (n={T})  [BH-corrected] ← primary result")
    _print_granger_table(gc_p1,   f"Sub-period 1: {p1_label}  [raw p — stability check]")
    _print_granger_table(gc_p2,   f"Sub-period 2: {p2_label}  [raw p — stability check]")

    # ── [4/5] Summary ────────────────────────────────────────────────────────
    print(f"\n[4/5]  SUMMARY")
    _print_summary(gc_full, gc_p1, gc_p2, p1_label, p2_label,
                   dAIC_tr, dBIC_tr, agr_tr, agr_te)

    # ── [5/5] Plots ──────────────────────────────────────────────────────────
    print(f"\n[5/5]  GENERATING PLOTS")
    _plot_granger_results(gc_full)
    _plot_model_comparison(tr, te,
                           aic_m_tr, aic_f_tr, bic_m_tr, bic_f_tr,
                           ll_m_te, ll_f_te, cm_tr, cm_te, agr_tr, agr_te)

    print(f"\nDone. Plots saved to {OUTPUT_DIR}/\n")
    return dict(df=df, tr=tr, te=te,
                hmm_market=hmm_m, hmm_fed=hmm_f,
                gc_full=gc_full,
                dAIC=dAIC_tr, dBIC=dBIC_tr, dLL_oos=dLL_te,
                agr_train=agr_tr, agr_test=agr_te)


# ═════════════════════════════════════════════════════════════════════════════
#  FORMATTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _header(title, subtitle=""):
    w = 72
    print(f"\n{'═' * w}\n  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'═' * w}")


def _print_granger_table(all_gc: dict, label: str = ""):
    n_tests = sum(len(v) for v in all_gc.values())
    w = 90
    print(f"\n       ── {label} ──")
    print(f"       {'─' * w}")
    print(f"       {'Cause':<18} {'Target':<18} {'Lag':>3}  "
          f"{'F-stat':>7}  {'p-raw':>7}  {'p-BH':>7}  {'Sig':>4}  {'ΔR²':>7}")
    print(f"       {'─' * w}")
    for (cause, tgt), rows in all_gc.items():
        for r in rows:
            p_adj = r.get("p_adj", r["p"])
            print(f"       {cause:<18} {tgt:<18} {r['lag']:>3}  "
                  f"{r['F']:>7.3f}  {r['p']:>7.4f}  {p_adj:>7.4f}  "
                  f"{stars(p_adj):>4}  {r['r2_gain']:>7.4f}")
        print(f"       {'·' * w}")
    print(f"       Key: ns=p≥.10  †=p<.10  *=p<.05  **=p<.01  ***=p<.001")
    print(f"       Sig based on BH-adjusted p  (m={n_tests} tests in this period)")


def _print_summary(gc_full, gc_p1, gc_p2, p1_label, p2_label,
                   dAIC, dBIC, agr_tr, agr_te):
    w = 80

    def best_result(gc, cause, use_adj=True):
        key = "p_adj" if use_adj else "p"
        ps = [(r.get(key, r["p"]), r["lag"])
              for r in gc.get((cause, "Rule Regime"), [])]
        return min(ps, key=lambda x: x[0]) if ps else (1.0, 0)

    print(f"\n       ╔{'═' * w}╗")
    print(f"       ║{'  RESULTS SUMMARY':^{w}}║")
    print(f"       ╠{'═' * w}╣")

    fed_ok = dAIC > 0 and dBIC > 0
    print(f"       ║  HMM Model Fit (in-sample):  ΔAIC={dAIC:+.1f}, ΔBIC={dBIC:+.1f}{' '*max(0,w-41)}║")
    print(f"       ║    → {'Fed IMPROVES in-sample fit ✓' if fed_ok else 'Fed does NOT improve in-sample fit':<{w-6}}║")
    print(f"       ╠{'═' * w}╣")
    print(f"       ║  Regime agreement (Rule ↔ HMM):  P1={agr_tr:.1%}  P2={agr_te:.1%}{' '*max(0,w-52)}║")
    print(f"       ╠{'═' * w}╣")
    print(f"       ║  Full sample: BH-adjusted p.  Sub-periods: raw p (stability){' '*max(0,w-63)}║")
    p1_short = p1_label.split('(')[0].strip()
    p2_short = p2_label.split('(')[0].strip()
    print(f"       ║  {'Cause':<22} {'Full (BH)':>14}  {p1_short:>14}  {p2_short:>14}  {'Stable?':>7}{'':>{max(0,w-77)}}║")
    print(f"       ║  {'─'*75}{' '*max(0,w-77)}║")
    for cause in CAUSE_COLS:
        p_full, l_full = best_result(gc_full, cause, use_adj=True)
        p_p1,   l_p1   = best_result(gc_p1,  cause, use_adj=False)
        p_p2,   l_p2   = best_result(gc_p2,  cause, use_adj=False)
        stable = ("YES ✓"  if p_p1 < 0.05 and p_p2 < 0.05 else
                  "P2 only" if p_p2 < 0.05 else
                  "P1 only" if p_p1 < 0.05 else "NO")
        line = (f"  {cause:<22} p={p_full:.3f}@{l_full}m{stars(p_full):>3}  "
                f"p={p_p1:.3f}@{l_p1}m{stars(p_p1):>3}  "
                f"p={p_p2:.3f}@{l_p2}m{stars(p_p2):>3}  {stable:>7}")
        print(f"       ║{line:<{w}}║")
    print(f"       ╠{'═' * w}╣")
    print(f"       ║  Stable? = raw p < .05 in BOTH sub-periods (exploratory, not a formal test){' '*max(0,w-79)}║")
    print(f"       ╚{'═' * w}╝\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Full-sample Conditional Granger Results (Rule Regime target)
# ═════════════════════════════════════════════════════════════════════════════
def _plot_granger_results(gc_full: dict):
    """
    3 panels (horizontal): F-statistics | p-values (BH) | ΔR².
    Target: rule-based bear regime (the only Granger target).
    """
    tgt = "Rule Regime"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle(
        "Conditional Granger Causality: Fed → Rule-based Bear Regime  [Full Sample]\n"
        f"Controls: {', '.join(CONTROL_COLS)}  |  BH-adjusted p-values  |  m=20 tests",
        color=TEXT, fontsize=11, fontweight="bold", y=1.02,
    )

    def style(ax, title):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

    x      = np.arange(len(LAGS))
    n_c    = len(CAUSE_COLS)
    w_bar  = 0.8 / n_c
    f_crit = stats.f.ppf(0.95, 1, 150)

    # Panel 0: F-statistics
    ax = axes[0]
    style(ax, "F-statistics")
    for i, cause in enumerate(CAUSE_COLS):
        rows = {r["lag"]: r for r in gc_full.get((cause, tgt), [])}
        Fs   = [rows.get(l, {}).get("F", 0) for l in LAGS]
        ax.bar(x + i * w_bar, Fs, w_bar * 0.9,
               color=CAUSE_COLORS[cause], alpha=0.85, label=cause)
    ax.axhline(f_crit, color="yellow", lw=1.2, ls="--", label=f"F crit≈{f_crit:.1f}")
    ax.set_xticks(x + w_bar * (n_c - 1) / 2)
    ax.set_xticklabels([f"lag {l}m" for l in LAGS], color=TEXT, fontsize=7)
    ax.set_ylabel("F-statistic")
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=6.5)

    # Panel 1: BH-adjusted p-values
    ax = axes[1]
    style(ax, "p-values (BH-adjusted)")
    for i, cause in enumerate(CAUSE_COLS):
        rows = {r["lag"]: r for r in gc_full.get((cause, tgt), [])}
        ps   = [rows.get(l, {}).get("p_adj", 1.0) for l in LAGS]
        ax.bar(x + i * w_bar, ps, w_bar * 0.9,
               color=CAUSE_COLORS[cause], alpha=0.85, label=cause)
    ax.axhline(0.05, color="yellow", lw=1.0, ls="--", label="p=.05")
    ax.axhline(0.10, color="yellow", lw=0.7, ls=":",  label="p=.10")
    ax.set_xticks(x + w_bar * (n_c - 1) / 2)
    ax.set_xticklabels([f"lag {l}m" for l in LAGS], color=TEXT, fontsize=7)
    ax.set_ylabel("p-value (BH adj)"); ax.set_ylim(0, 1.05)
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=6.5)

    # Panel 2: ΔR²
    ax = axes[2]
    style(ax, "ΔR² gain by lag")
    for cause in CAUSE_COLS:
        rows  = sorted(gc_full.get((cause, tgt), []), key=lambda r: r["lag"])
        lgs   = [r["lag"]     for r in rows]
        gains = [r["r2_gain"] for r in rows]
        ax.plot(lgs, gains, "o-", color=CAUSE_COLORS[cause],
                lw=2, ms=6, label=cause)
    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.3)
    ax.set_xlabel("Lag (months)"); ax.set_ylabel("ΔR²")
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=6.5)

    out = os.path.join(OUTPUT_DIR, "2_conditional_granger.png")
    fig.tight_layout(pad=2.0)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Model Comparison & Regime Distributions
# ═════════════════════════════════════════════════════════════════════════════
def _plot_model_comparison(tr, te,
                           aic_m_tr, aic_f_tr, bic_m_tr, bic_f_tr,
                           ll_m_te, ll_f_te, cm_tr, cm_te, agr_tr, agr_te):
    dAIC = aic_m_tr - aic_f_tr
    dBIC = bic_m_tr - bic_f_tr
    dLL_oos = ll_f_te - ll_m_te

    fig = plt.figure(figsize=(17, 12), facecolor=BG)
    fig.suptitle("HMM vs Rule-based Regime Comparison  (train left | test right)",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.35,
                           left=0.07, right=0.97, top=0.95, bottom=0.05)

    def style(ax, title):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=TEXT, fontsize=8.5, fontweight="bold", pad=5)
        ax.tick_params(colors=TEXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#30363D")
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

    # [0,0] AIC/BIC (in-sample) + OOS LL comparison
    ax = fig.add_subplot(gs[0, 0])
    style(ax, f"In-sample Fit  ΔAIC={dAIC:+.1f}  ΔBIC={dBIC:+.1f}\n"
             f"OOS ΔLL(fed−mkt)={dLL_oos:+.2f}")
    cats   = ["AIC\nMkt", "AIC\n+Fed", "BIC\nMkt", "BIC\n+Fed"]
    vals   = [aic_m_tr, aic_f_tr, bic_m_tr, bic_f_tr]
    colors_ = [NEUT, FED, NEUT, FED]
    bars   = ax.bar(cats, vals, color=colors_, alpha=0.85, edgecolor="#30363D", width=0.55)
    base   = min(vals) * 0.998
    ax.set_ylim(base, max(vals) * 1.015)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (max(vals) - base) * 0.003,
                f"{v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=7)
    ax.legend(handles=[Patch(color=NEUT, label="Market-only"),
                       Patch(color=FED,  label="+Fed features")],
              facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # [0,1] Confusion matrix — TRAIN
    ax = fig.add_subplot(gs[0, 1])
    style(ax, f"TRAIN Agreement: {agr_tr:.1%}\n(rows=Rule, cols=HMM Viterbi)")
    im = ax.imshow(cm_tr, cmap="RdYlGn", vmin=0, vmax=cm_tr.max(), aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_tr[i,j]}", ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HMM Bull", "HMM Bear"], color=TEXT)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Rule Bull", "Rule Bear"], color=TEXT)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # [0,2] Confusion matrix — TEST (OOS)
    ax = fig.add_subplot(gs[0, 2])
    style(ax, f"TEST Agreement: {agr_te:.1%}  (out-of-sample)\n(rows=Rule, cols=HMM Viterbi)")
    im2 = ax.imshow(cm_te, cmap="RdYlGn", vmin=0, vmax=cm_te.max(), aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_te[i,j]}", ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HMM Bull", "HMM Bear"], color=TEXT)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Rule Bull", "Rule Bear"], color=TEXT)
    plt.colorbar(im2, ax=ax, fraction=0.045, pad=0.04)

    # Row 1: Distributions by regime — Fed_Cycle (train), Fed_Cycle (OOS), SPX_Return (OOS)
    for col_i, (col, title, seg) in enumerate([
        ("Fed_Cycle",   "Fed Cycle (TRAIN)",     tr),
        ("Fed_Cycle",   "Fed Cycle (TEST/OOS)",  te),
        ("SPX_Return",  "SPX Return (TEST/OOS)", te),
    ]):
        ax = fig.add_subplot(gs[1, col_i])
        style(ax, f"{title}\n(solid=Rule-based, dashed=HMM)")
        rule_col = "bear_rule"
        hmm_col  = "hmm_state"
        for state, c, name in [(0, BULL, "Bull"), (1, BEAR, "Bear")]:
            sub_r = seg[seg[rule_col] == state][col].dropna()
            sub_h = seg[seg[hmm_col]  == state][col].dropna()
            if len(sub_r) > 5:
                sub_r.plot.kde(ax=ax, color=c, lw=2.0, ls="-",
                               label=f"Rule {name}  μ={sub_r.mean():.2f}")
            if len(sub_h) > 5:
                sub_h.plot.kde(ax=ax, color=c, lw=1.4, ls="--", alpha=0.7,
                               label=f"HMM  {name}  μ={sub_h.mean():.2f}")
        ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=6.5)
        ax.set_xlabel(col)

    out = os.path.join(OUTPUT_DIR, "3_model_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(full)
