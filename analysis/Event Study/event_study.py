"""
Event Study & Lead-Lag Cross-Correlation Analysis
==================================================
Tests whether Federal Reserve rate hikes are a TRIGGER (leading indicator)
or a REACTION (lagging indicator) to bull-to-bear S&P 500 regime switches.

Two complementary approaches:
  1. Event Study   — average market behaviour around each hike cycle start
  2. Lead-Lag CCF  — cross-correlation at lags -24m to +24m (which leads which?)

Run from the project root:
    python "analysis/Event Study/event_study.py"
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.scripts import data_generation
from data.scripts.analysis_utils import classify_regimes

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
START_DATE   = "2000-01-01"
END_DATE     = "2026-01-01"
WINDOW_PRE   = 12   # months before hike cycle start to include
WINDOW_POST  = 24   # months after hike cycle start to include
LAGS         = list(range(-24, 25))

# ── Colours (consistent with granger_causality.py) ───────────────────────────
BG   = "#0D1117"; CARD = "#161B22"; TEXT = "#C9D1D9"
BULL = "#2ECC71"; BEAR = "#E74C3C"; FED  = "#3498DB"
NEUT = "#95A5A6"; GOLD = "#F39C12"


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def _build_dataset() -> pd.DataFrame:
    technical = data_generation.get_yahoo_finance_data(START_DATE, END_DATE)
    macro     = data_generation.get_fred_input_data(START_DATE, END_DATE)
    df        = pd.concat([technical, macro], axis=1)

    # bear_rule: 1 = Bear, 0 = Bull  (flipped from classify_regimes which returns 1=Bull)
    df["bear_rule"]  = (1 - classify_regimes(df)).astype(int)
    df["SPX_Return"] = np.log(df["SPX_Close"]).diff()

    if "Fed_Funds_Rate" in df.columns:
        df["Fed_Change"] = df["Fed_Funds_Rate"].diff()
        df["Fed_Cycle"]  = _compute_fed_cycle(df["Fed_Change"])

    df.dropna(subset=["SPX_Close", "SPX_Return", "Fed_Funds_Rate",
                      "Fed_Change", "bear_rule"], inplace=True)
    return df


def _compute_fed_cycle(fed_change: pd.Series, threshold: float = 0.05) -> pd.Series:
    """Persistent hike/cut direction: +1 = hiking cycle, -1 = cutting cycle."""
    cycle, last = pd.Series(0.0, index=fed_change.index), 0.0
    for i, v in enumerate(fed_change.values):
        if v > threshold:    last = 1.0
        elif v < -threshold: last = -1.0
        cycle.iloc[i] = last
    return cycle


# ═════════════════════════════════════════════════════════════════════════════
#  HIKE CYCLE IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════
def identify_hike_cycle_starts(fed_change: pd.Series,
                               pause_months: int = 3) -> pd.DatetimeIndex:
    """
    Detect the FIRST hike of each distinct tightening cycle.
    A cycle start = month with Fed_Change > 0 after >= pause_months months
    without a hike. A cut (Fed_Change < 0) ends the current cycle.
    """
    starts       = []
    in_hike      = False
    flat_count   = 0

    for date, val in fed_change.items():
        if val > 0:
            if not in_hike and flat_count >= pause_months:
                starts.append(date)
            in_hike    = True
            flat_count = 0
        else:
            flat_count += 1
            if val < 0:   # rate cut ends the cycle
                in_hike = False

    return pd.DatetimeIndex(starts)


# ═════════════════════════════════════════════════════════════════════════════
#  1) EVENT STUDY
# ═════════════════════════════════════════════════════════════════════════════
def run_event_study(df: pd.DataFrame,
                    hike_starts: pd.DatetimeIndex) -> dict:
    """
    For each hike cycle start, extract cumulative SPX log-return and bear
    regime flag across the window [-WINDOW_PRE, +WINDOW_POST] months.
    Returns are normalised to 0 at t=0 (the hike start month).
    """
    window     = list(range(-WINDOW_PRE, WINDOW_POST + 1))
    all_dates  = df.index

    cum_returns  = []
    bear_flags   = []
    event_labels = []

    for start_date in hike_starts:
        # Snap to nearest available date
        idx = all_dates.searchsorted(start_date)
        if idx >= len(all_dates):
            continue
        start_date = all_dates[idx]
        event_idx  = all_dates.get_loc(start_date)

        # Need full window on both sides
        if event_idx - WINDOW_PRE < 0 or event_idx + WINDOW_POST >= len(all_dates):
            continue

        win_dates = all_dates[event_idx - WINDOW_PRE : event_idx + WINDOW_POST + 1]

        # Cumulative log-return anchored to 0 at t=0
        log_px  = np.log(df.loc[win_dates, "SPX_Close"].values.astype(float))
        cum_ret = log_px - log_px[WINDOW_PRE]

        bear = df.loc[win_dates, "bear_rule"].values

        cum_returns.append(pd.Series(cum_ret, index=window))
        bear_flags.append(pd.Series(bear.astype(float), index=window))
        event_labels.append(start_date.strftime("%Y-%m"))

    return {
        "cum_returns": pd.DataFrame(cum_returns, index=event_labels),
        "bear_flags":  pd.DataFrame(bear_flags,  index=event_labels),
        "window":      window,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  2) LEAD-LAG CROSS-CORRELATION
# ═════════════════════════════════════════════════════════════════════════════
def compute_lead_lag_ccf(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Fed variable and each lag L in [-24, +24], compute:
        CCF(L) = corr( Fed[t],  Bear[t + L] )

    L > 0 → Fed at time t is correlated with bear regime L months LATER
            → if peak is at positive L: Fed LEADS bear (hikes may be a trigger)
    L < 0 → Fed at time t is correlated with bear regime |L| months EARLIER
            → if peak is at negative L: bear regime LEADS Fed (market falls first)
    """
    results = {}
    for var in ["Fed_Funds_Rate", "Fed_Change", "Fed_Cycle"]:
        corrs = [
            df[var].corr(df["bear_rule"].shift(-lag))
            for lag in LAGS
        ]
        results[var] = pd.Series(corrs, index=LAGS)

    return pd.DataFrame(results)


def _sig_threshold(n: int, alpha: float = 0.05) -> float:
    """95% CI half-width for zero-correlation null (Bartlett approximation)."""
    return stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def _style(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(colors=TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363D")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def plot_event_study(es: dict, save_path: str):
    cum_ret  = es["cum_returns"]
    bear_pct = es["bear_flags"].mean()   # fraction of episodes in bear at each window point
    window   = es["window"]
    n_events = len(cum_ret)

    mean_ret = cum_ret.mean()
    std_ret  = cum_ret.std()
    x        = np.array(window)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        f"Event Study: S&P 500 Around Fed Hike Cycle Starts  ({n_events} cycles, 2000–2025)\n"
        "t = 0 marks the first rate hike of each cycle",
        color=TEXT, fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    # [0,0] Spaghetti plot: individual episode returns + average
    ax = fig.add_subplot(gs[0, 0])
    _style(ax, "Individual Episode Cumulative Returns")
    for label, row in cum_ret.iterrows():
        ax.plot(window, row.values * 100, color=NEUT, lw=0.8, alpha=0.35)
    ax.plot(x, mean_ret.values * 100, color=FED, lw=2.5, label="Average", zorder=5)
    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
    ax.axvline(0, color=GOLD, lw=1.2, ls="--", alpha=0.8, label="Hike starts (t=0)")
    ax.set_xlabel("Months relative to hike cycle start")
    ax.set_ylabel("Cumulative log return (%)")
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # [0,1] Average return ± 1 std band
    ax = fig.add_subplot(gs[0, 1])
    _style(ax, "Average Cumulative Return  ± 1 Std Dev")
    ax.fill_between(x,
                    (mean_ret - std_ret).values * 100,
                    (mean_ret + std_ret).values * 100,
                    color=FED, alpha=0.2, label="±1 std")
    ax.plot(x, mean_ret.values * 100, color=FED, lw=2.5, label="Average")
    ax.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
    ax.axvline(0, color=GOLD, lw=1.2, ls="--", alpha=0.8, label="Hike starts (t=0)")
    ax.set_xlabel("Months relative to hike cycle start")
    ax.set_ylabel("Cumulative log return (%)")
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # [1,0] Bear market frequency across the window
    ax = fig.add_subplot(gs[1, 0])
    _style(ax, "Bear Market Frequency by Window Position")
    bar_colors = [BEAR if v > 0.4 else BULL for v in bear_pct.values]
    ax.bar(window, bear_pct.values * 100, color=bar_colors, alpha=0.75, width=0.85)
    ax.axhline(40, color=TEXT, lw=0.8, ls=":", alpha=0.5, label="40% reference")
    ax.axvline(0, color=GOLD, lw=1.2, ls="--", alpha=0.8, label="Hike starts (t=0)")
    ax.set_xlabel("Months relative to hike cycle start")
    ax.set_ylabel("% of episodes in Bear market")
    ax.set_ylim(0, 105)
    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=7)

    # [1,1] Pre / at / post summary bar chart
    ax = fig.add_subplot(gs[1, 1])
    _style(ax, "Bear Market Rate: Pre vs. Post Hike Start")
    pre_range    = [w for w in window if -WINDOW_PRE <= w < 0]
    post_range   = [w for w in window if 0 < w <= WINDOW_POST]

    pre_bear  = bear_pct[pre_range].mean()  * 100
    at_bear   = bear_pct[0]                 * 100
    post_bear = bear_pct[post_range].mean() * 100

    # Split post into early (1-12m) and late (13-24m) for more granularity
    early_post = [w for w in window if 0 < w <= 12]
    late_post  = [w for w in window if 12 < w <= 24]
    early_bear = bear_pct[early_post].mean() * 100 if early_post else 0
    late_bear  = bear_pct[late_post].mean()  * 100 if late_post  else 0

    cats   = ["Pre-hike\n(avg 12m)", "At hike\nstart", "Post 1-12m\n(avg)", "Post 13-24m\n(avg)"]
    vals   = [pre_bear, at_bear, early_bear, late_bear]
    colors = [BULL if v < 40 else BEAR for v in vals]
    bars   = ax.bar(cats, vals, color=colors, alpha=0.85, edgecolor="#30363D", width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")
    ax.set_ylabel("% of episodes in Bear market")
    ax.set_ylim(0, 110)
    ax.legend(handles=[Patch(color=BULL, label="Mostly Bull (<40%)"),
                       Patch(color=BEAR, label="Mostly Bear (≥40%)")],
              facecolor=CARD, labelcolor=TEXT, fontsize=7)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {save_path}")


def plot_lead_lag(ccf_df: pd.DataFrame, n_obs: int, save_path: str):
    sig = _sig_threshold(n_obs)

    col_colors = {
        "Fed_Funds_Rate": FED,
        "Fed_Change":     GOLD,
        "Fed_Cycle":      "#9B59B6",
    }
    col_labels = {
        "Fed_Funds_Rate": "Level of Policy Rate",
        "Fed_Change":     "Month-over-Month Change",
        "Fed_Cycle":      "Persistent Hike/Cut Direction",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=BG)
    fig.suptitle(
        "Lead-Lag Cross-Correlation: Fed Variables  ↔  Bear Regime\n"
        "CCF(L) = corr(Fed[t], Bear[t+L])  |  "
        "L > 0: Fed leads Bear  |  L < 0: Bear leads Fed",
        color=TEXT, fontsize=11, fontweight="bold", y=1.03,
    )

    for ax, col in zip(axes, ccf_df.columns):
        _style(ax, col_labels[col])
        lags  = ccf_df.index.values
        corrs = ccf_df[col].values

        bar_colors = [col_colors[col] if c > 0 else BEAR for c in corrs]
        ax.bar(lags, corrs, color=bar_colors, alpha=0.70, width=0.85)

        # Significance bands
        ax.axhline( sig, color="yellow", lw=1.0, ls="--", alpha=0.8,
                   label=f"95% CI (±{sig:.2f})")
        ax.axhline(-sig, color="yellow", lw=1.0, ls="--", alpha=0.8)
        ax.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
        ax.axvline(0, color=GOLD, lw=1.0, ls=":", alpha=0.6, label="Lag 0")

        # Mark the peak absolute correlation
        peak_idx = int(np.argmax(np.abs(corrs)))
        peak_lag = lags[peak_idx]
        peak_val = corrs[peak_idx]
        direction = "Fed leads" if peak_lag > 0 else ("Bear leads" if peak_lag < 0 else "contemporaneous")
        ax.annotate(
            f"Peak  lag={peak_lag:+d}m\nr={peak_val:.2f}\n({direction})",
            xy=(peak_lag, peak_val),
            xytext=(peak_lag + (5 if peak_lag < 15 else -15),
                    peak_val + (0.08 if peak_val > 0 else -0.12)),
            color=TEXT, fontsize=6.5,
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=0.8),
        )

        ax.set_xlabel("Lag L (months)  [positive = Fed leads Bear]")
        ax.set_ylabel("Correlation coefficient")
        ax.set_xlim(min(lags) - 1, max(lags) + 1)
        ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=6.5)

    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"       Saved → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════
def _print_event_study_summary(es: dict):
    cum_ret  = es["cum_returns"]
    bear_pct = es["bear_flags"].mean()
    window   = es["window"]

    print(f"\n       Hike cycles included in event study:")
    for label in cum_ret.index:
        t0_bear = es["bear_flags"].loc[label, 0]
        t12_bear = es["bear_flags"].loc[label, 12] if 12 in window else "N/A"
        print(f"         {label}  |  bear at t=0: {'Yes' if t0_bear else 'No '}  "
              f"|  bear at t+12m: {'Yes' if t12_bear else 'No '}")

    print(f"\n       Average cumulative SPX return by window:")
    for t in [-12, -6, -3, 0, 3, 6, 12, 18, 24]:
        if t in window:
            r = cum_ret[t].mean() * 100
            b = bear_pct[t] * 100
            flag = "  ← hike starts" if t == 0 else ""
            print(f"         t={t:+3d}m  avg return = {r:+5.1f}%  |  bear freq = {b:.0f}%{flag}")

    pre_range  = [w for w in window if w < 0]
    post_range = [w for w in window if w > 0]
    pre_bear   = bear_pct[pre_range].mean()  * 100
    post_bear  = bear_pct[post_range].mean() * 100

    print(f"\n       Pre-hike  avg bear frequency:  {pre_bear:.1f}%")
    print(f"       Post-hike avg bear frequency: {post_bear:.1f}%")
    diff = post_bear - pre_bear

    if diff > 15:
        verdict = "Bear markets become MORE common after hike starts → hikes may be a trigger."
    elif diff < -10:
        verdict = "Bear markets are MORE common BEFORE hike starts → markets fall first, hikes follow."
    else:
        verdict = "Bear frequency is similar pre/post hike → no clear directional timing effect."
    print(f"\n       FINDING: {verdict}")


def _print_lead_lag_summary(ccf_df: pd.DataFrame, n_obs: int):
    sig = _sig_threshold(n_obs)
    w   = 72
    print(f"\n       Significance threshold (95% CI): ±{sig:.3f}  (n={n_obs})")
    print(f"\n       {'Variable':<22} {'Peak lag':>10}  {'Peak r':>8}  {'Direction':>22}  {'Sig lags'}")
    print(f"       {'─'*w}")

    for col in ccf_df.columns:
        corrs    = ccf_df[col].values
        lags     = ccf_df.index.values
        peak_idx = int(np.argmax(np.abs(corrs)))
        peak_lag = lags[peak_idx]
        peak_val = corrs[peak_idx]
        sig_lags = sorted(lags[np.abs(corrs) > sig].tolist())

        if peak_lag > 3:
            direction = f"Fed leads Bear  (+{peak_lag}m)"
        elif peak_lag < -3:
            direction = f"Bear leads Fed  ({peak_lag}m)"
        else:
            direction = "Contemporaneous"

        print(f"       {col:<22} {peak_lag:>+10d}  {peak_val:>+8.3f}  {direction:>22}  {sig_lags}")

    print(f"\n       Interpretation guide:")
    print(f"         Peak at large positive lag → Fed hikes CAUSE regime switch (trigger hypothesis)")
    print(f"         Peak at large negative lag → regime switch PRECEDES Fed action (reaction hypothesis)")
    print(f"         Peak near zero            → contemporaneous: hikes and bear markets coincide")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    w = 72
    print(f"\n{'═'*w}\n  EVENT STUDY & LEAD-LAG ANALYSIS\n{'═'*w}")

    # [1/4] Data
    print("\n[1/4]  Loading data...")
    df = _build_dataset()
    print(f"       {len(df)} monthly observations: "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    # [2/4] Event study
    print("\n[2/4]  Event study...")
    hike_starts = identify_hike_cycle_starts(df["Fed_Change"], pause_months=3)
    print(f"       Identified {len(hike_starts)} hike cycle starts: "
          f"{[d.strftime('%Y-%m') for d in hike_starts]}")

    es = run_event_study(df, hike_starts)
    _print_event_study_summary(es)

    # [3/4] Lead-lag CCF
    print("\n[3/4]  Lead-lag cross-correlation...")
    ccf_df = compute_lead_lag_ccf(df)
    _print_lead_lag_summary(ccf_df, n_obs=len(df))

    # [4/4] Plots
    print("\n[4/4]  Generating plots...")
    plot_event_study(es, os.path.join(OUTPUT_DIR, "1_event_study.png"))
    plot_lead_lag(ccf_df, len(df), os.path.join(OUTPUT_DIR, "2_lead_lag_ccf.png"))

    print(f"\n{'═'*w}")
    print(f"  Done. Plots saved to: {OUTPUT_DIR}/")
    print(f"{'═'*w}\n")

    return dict(df=df, es=es, ccf_df=ccf_df, hike_starts=hike_starts)


if __name__ == "__main__":
    run()
