"""
NQ Stacked Volume Profile Analyzer
===================================
Builds 4 VP windows (weekly / 30d / 60d / 90d) and stacks their structural
features (VAH, VAL, POC, HVN shelves, HVN ledges, LVNs) to identify
high-probability reversal zones.

Usage
-----
  python stacked_vp.py                  # rolling mode (default)
  python stacked_vp.py --anchored       # calendar-anchored profiles
  python stacked_vp.py --compare        # run both and compare
  python stacked_vp.py --bins 2.5       # bin size in NQ points (default 5)
  python stacked_vp.py --display 1000   # recent bars in chart (default 600)
  python stacked_vp.py --top 25         # print top N zones (default 20)

Shelf  = HVN boundary with gradual volume taper  (gentle gradient)
Ledge  = HVN boundary with abrupt volume cliff   (steep gradient)
"""

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Ensure UTF-8 output on Windows (box-drawing chars, arrows in comments/labels)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("hdata")
CSV_PATH   = DATA_DIR / "NQ_1m_clean.csv"
CACHE_PATH = DATA_DIR / "NQ_1m_clean_cache.parquet"

# ─── Parameters ───────────────────────────────────────────────────────────────
BIN_SIZE         = 5.0    # NQ points per bin (tick = 0.25; 5 pts = ~20 ticks)
VALUE_AREA_PCT   = 0.70   # fraction of total volume that defines the VA
HVN_PROMINENCE   = 0.18   # HVN peak must exceed this × max_volume in prominence
LVN_DEPTH        = 0.50   # LVN valley must be < this × mean volume
SHELF_SLOPE_MAX  = 0.08   # normalised avg slope ≤ this  →  shelf (gentle taper)
LEDGE_SLOPE_MIN  = 0.25   # normalised avg slope ≥ this  →  ledge (sharp cliff)
SLOPE_BINS       = 5      # bins used to measure boundary gradient
STACK_TOLERANCE  = 1.5    # confluence radius = this × BIN_SIZE
DISPLAY_BARS     = 600    # recent 1-min bars shown in price chart

LOOKBACKS = {             # approximate trading days per window
    "weekly": 5,
    "30d":    21,
    "60d":    42,
    "90d":    63,
}
MINS_PER_DAY = 390        # ~6.5 h × 60


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    name:  str
    bins:  np.ndarray   # left edge of each price bin
    vols:  np.ndarray   # volume assigned to each bin
    poc:   float        # point of control (highest-volume bin)
    vah:   float        # value area high
    val:   float        # value area low


@dataclass
class VPFeature:
    price:     float
    ftype:     str      # vah | val | poc | shelf | ledge | lvn
    timeframe: str
    weight:    float = 1.0


@dataclass
class Zone:
    price:     float
    score:     float    # confluence strength  (n_tf² × Σ weights)
    n_tf:      int      # unique timeframes contributing
    timeframes: set
    ftypes:    set
    features:  list


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"  cache  → {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        # Normalise: the richer cache uses 'ts' for the timestamp column
        if "date" not in df.columns and "ts" in df.columns:
            df = df.rename(columns={"ts": "date"})
        elif "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    else:
        print(f"  csv    → {CSV_PATH}  (first load — caching to parquet)")
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(CACHE_PATH)
    # Keep only the columns we need, apply 2020 cutoff
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2020-01-01"]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── Volume Profile Construction ──────────────────────────────────────────────

def build_profile(bars: pd.DataFrame, name: str, bin_size: float) -> VolumeProfile:
    """
    Distribute each bar's volume uniformly across price bins spanning [low, high].
    This is the standard OHLCV approximation for volume-at-price.
    """
    lo_arr  = bars["low"].values
    hi_arr  = bars["high"].values
    vol_arr = bars["volume"].values.astype(np.float64)

    price_min = np.floor(lo_arr.min() / bin_size) * bin_size
    price_max = np.ceil(hi_arr.max()  / bin_size) * bin_size
    n_bins    = int(round((price_max - price_min) / bin_size)) + 1
    profile   = np.zeros(n_bins, dtype=np.float64)

    lo_idx = np.floor((lo_arr - price_min) / bin_size).astype(int).clip(0, n_bins - 1)
    hi_idx = np.floor((hi_arr - price_min) / bin_size).astype(int).clip(0, n_bins - 1)

    for i in range(len(bars)):
        span = hi_idx[i] - lo_idx[i] + 1
        profile[lo_idx[i]: hi_idx[i] + 1] += vol_arr[i] / span

    bins = price_min + np.arange(n_bins) * bin_size

    # Point of Control
    poc = bins[np.argmax(profile)]

    # Value Area: greedily add highest-volume bins until 70 % threshold
    total     = profile.sum()
    order     = np.argsort(profile)[::-1]
    cumsum    = 0.0
    va_idx    = set()
    for idx in order:
        cumsum += profile[idx]
        va_idx.add(int(idx))
        if cumsum >= total * VALUE_AREA_PCT:
            break
    va_sorted = sorted(va_idx)
    vah = bins[va_sorted[-1]] + bin_size  # top of the highest VA bin
    val = bins[va_sorted[0]]

    return VolumeProfile(name=name, bins=bins, vols=profile, poc=poc, vah=vah, val=val)


# ─── Feature Detection ────────────────────────────────────────────────────────

def detect_features(vp: VolumeProfile,
                    shelf_max: float = None,
                    ledge_min: float = None) -> list:
    shelf_max = SHELF_SLOPE_MAX if shelf_max is None else shelf_max
    ledge_min = LEDGE_SLOPE_MIN if ledge_min is None else ledge_min

    feats: list[VPFeature] = []
    bins, vols = vp.bins, vp.vols
    tf = vp.name

    feats += [
        VPFeature(vp.vah, "vah", tf, weight=1.2),
        VPFeature(vp.val, "val", tf, weight=1.2),
    ]

    # Smooth the raw profile before structural analysis
    n   = len(vols)
    win = max(5, n // 15)
    if win % 2 == 0:
        win += 1
    win = min(win, n - 1 if n % 2 == 0 else n)
    try:
        smooth = savgol_filter(vols, window_length=win, polyorder=2).clip(0)
    except Exception:
        smooth = vols.copy()

    peak_vol = smooth.max()
    mean_vol = smooth.mean()
    if peak_vol < 1e-9:
        return feats

    # ── HVN: peaks with significant prominence ────────────────────────────────
    peaks, _ = find_peaks(smooth, prominence=HVN_PROMINENCE * peak_vol)

    for p in peaks:
        try:
            _, _, lo_ips, hi_ips = peak_widths(smooth, [p], rel_height=0.5)
        except Exception:
            continue
        l_idx = int(np.clip(lo_ips[0], 0, n - 1))
        r_idx = int(np.clip(hi_ips[0], 0, n - 1))

        _classify_boundary(smooth, bins, l_idx, "lower", peak_vol, tf, feats,
                            shelf_max, ledge_min)
        _classify_boundary(smooth, bins, r_idx, "upper", peak_vol, tf, feats,
                            shelf_max, ledge_min)

    # ── LVN: valleys significantly below mean ─────────────────────────────────
    valleys, _ = find_peaks(-smooth, prominence=0.08 * peak_vol)
    for v in valleys:
        if smooth[v] < LVN_DEPTH * mean_vol:
            feats.append(VPFeature(bins[v], "lvn", tf, weight=0.9))

    return feats


def _classify_boundary(smooth, bins, edge_idx, side, peak_vol, tf, feats,
                        shelf_max, ledge_min):
    """
    Measure the average normalised volume slope moving outward from an HVN edge.
    slope ≤ shelf_max  →  Shelf (gradual taper)
    slope ≥ ledge_min  →  Ledge (abrupt cliff)
    """
    n = len(smooth)
    if side == "lower":
        indices = list(range(edge_idx, max(-1, edge_idx - SLOPE_BINS - 1), -1))
    else:
        indices = list(range(edge_idx, min(n, edge_idx + SLOPE_BINS + 1)))

    seg = smooth[indices]
    if len(seg) < 2:
        return

    drops     = np.diff(seg)
    avg_slope = np.mean(np.abs(drops)) / peak_vol

    price = bins[edge_idx]
    if avg_slope <= shelf_max:
        feats.append(VPFeature(price, "shelf", tf, weight=0.8))
    elif avg_slope >= ledge_min:
        feats.append(VPFeature(price, "ledge", tf, weight=1.3))


# ─── Confluence Stacking ──────────────────────────────────────────────────────

def stack_features(all_features: list, bin_size: float) -> list:
    """
    Greedy clustering: group features within STACK_TOLERANCE × bin_size of
    each other.  Score = n_unique_timeframes² × Σ feature_weights.

    The n_tf² term heavily rewards multi-timeframe alignment — the core of the
    'stacked VP' concept.
    """
    if not all_features:
        return []

    tolerance = STACK_TOLERANCE * bin_size
    prices    = np.array([f.price for f in all_features])
    sort_idx  = np.argsort(prices)
    used      = np.zeros(len(all_features), dtype=bool)
    zones: list[Zone] = []

    for i in sort_idx:
        if used[i]:
            continue
        cluster = [all_features[i]]
        used[i] = True
        for j in sort_idx:
            if used[j]:
                continue
            if prices[j] > prices[i] + tolerance:
                break                    # sorted → no more matches
            if abs(prices[j] - prices[i]) <= tolerance:
                cluster.append(all_features[j])
                used[j] = True

        center = float(np.mean([f.price for f in cluster]))
        tfs    = {f.timeframe for f in cluster}
        ftypes = {f.ftype    for f in cluster}
        n_tf   = len(tfs)
        score  = (n_tf ** 2) * sum(f.weight for f in cluster)

        zones.append(Zone(
            price=center, score=score,
            n_tf=n_tf, timeframes=tfs, ftypes=ftypes, features=cluster,
        ))

    return sorted(zones, key=lambda z: z.score, reverse=True)


# ─── Lookback Cutoff ──────────────────────────────────────────────────────────

def get_cutoff(df: pd.DataFrame, name: str, anchored: bool) -> pd.Timestamp:
    last = df["date"].max()
    td   = LOOKBACKS[name]
    if not anchored:
        idx = max(0, len(df) - td * MINS_PER_DAY)
        return df.iloc[idx]["date"]
    # Anchored: calendar period boundaries
    if name == "weekly":
        return last - pd.Timedelta(days=last.weekday())
    months_back = {"30d": 1, "60d": 2, "90d": 3}[name]
    m, y = last.month - months_back, last.year
    while m <= 0:
        m += 12; y -= 1
    return pd.Timestamp(y, m, 1)


# ─── Analysis Driver ──────────────────────────────────────────────────────────

def run(df: pd.DataFrame, anchored: bool, bin_size: float):
    all_features: list[VPFeature] = []
    profiles: dict[str, VolumeProfile] = {}

    for name in LOOKBACKS:
        cutoff = get_cutoff(df, name, anchored)
        subset = df[df["date"] >= cutoff]
        if len(subset) < 20:
            print(f"  [{name}] insufficient data, skipping")
            continue

        vp = build_profile(subset, name, bin_size)
        profiles[name] = vp
        feats = detect_features(vp)
        all_features.extend(feats)

        print(f"  [{name}]  bars={len(subset):>7,}  "
              f"VAH={vp.vah:>9.2f}  VAL={vp.val:>9.2f}  "
              f"feats={len(feats)}")

    zones = stack_features(all_features, bin_size)
    return zones, profiles


# ─── Console Report ───────────────────────────────────────────────────────────

def print_report(zones: list, show_n: int = 20):
    sep = "-" * 74
    print(f"\n{sep}")
    print(f"  TOP {min(show_n, len(zones))} CONFLUENCE ZONES")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>6}  {'TF':>2}  TIMEFRAMES            FEATURE TYPES")
    print(sep)
    for z in zones[:show_n]:
        tfs = ",".join(sorted(z.timeframes))
        fts = "|".join(sorted(z.ftypes))
        print(f"  {z.price:>9.2f}  {z.score:>6.1f}  {z.n_tf:>2}  {tfs:<20}  {fts}")
    print(sep)
    print(f"  3+ timeframe zones : {sum(1 for z in zones if z.n_tf >= 3)}")
    print(f"  4  timeframe zones : {sum(1 for z in zones if z.n_tf == 4)}")


# ─── Visualization ────────────────────────────────────────────────────────────

TF_COLORS = {
    "weekly": "#FF6B6B",
    "30d":    "#FFA042",
    "60d":    "#FFD966",
    "90d":    "#4EC9B0",
}


def _draw_ohlc(ax, window: pd.DataFrame):
    """Render OHLC candlestick bars using LineCollections (fast, no extra deps)."""
    n = len(window)
    x = np.arange(n)
    o = window["open"].values
    h = window["high"].values
    l = window["low"].values
    c = window["close"].values
    up = c >= o

    # Wicks: one thin line per bar, low → high
    wick_segs = [[(i, l[i]), (i, h[i])] for i in range(n)]
    wick_cols = ["#3fb950" if up[i] else "#f85149" for i in range(n)]
    ax.add_collection(LineCollection(wick_segs, colors=wick_cols,
                                     linewidths=0.55, zorder=1))

    # Bodies: thick line open → close, coloured by direction
    up_segs = [[(i, o[i]), (i, c[i])] for i in range(n) if up[i]]
    dn_segs = [[(i, c[i]), (i, o[i])] for i in range(n) if not up[i]]
    if up_segs:
        ax.add_collection(LineCollection(up_segs, colors="#3fb950",
                                         linewidths=3.0, zorder=2))
    if dn_segs:
        ax.add_collection(LineCollection(dn_segs, colors="#f85149",
                                         linewidths=3.0, zorder=2))


def plot(df: pd.DataFrame, zones: list, profiles: dict,
         anchored: bool, bin_size: float, show_n: int = 15,
         random_window: bool = False):

    # ── Select display window ─────────────────────────────────────────────────
    if random_window:
        # Only pick windows whose price range overlaps with at least one zone.
        # Retry up to 200 times, then fall back to latest bars.
        zone_prices = np.array([z.price for z in zones]) if zones else np.array([])
        max_start   = max(0, len(df) - DISPLAY_BARS - 1)
        window      = None
        for _ in range(200):
            si   = int(np.random.randint(0, max_start + 1))
            seg  = df.iloc[si: si + DISPLAY_BARS]
            lo_w = seg["low"].min()
            hi_w = seg["high"].max()
            if len(zone_prices) == 0 or np.any((zone_prices >= lo_w) & (zone_prices <= hi_w)):
                window = seg.copy().reset_index(drop=True)
                break
        if window is None:
            window = df.tail(DISPLAY_BARS).copy().reset_index(drop=True)
    else:
        window = df.tail(DISPLAY_BARS).copy().reset_index(drop=True)

    date_start = window["date"].iloc[0]
    date_end   = window["date"].iloc[-1]

    price_lo = window["low"].min()
    price_hi = window["high"].max()
    spread   = price_hi - price_lo
    pad      = spread * 0.06
    y_lo, y_hi = price_lo - pad, price_hi + pad

    dark = "#0D1117"
    fig  = plt.figure(figsize=(22, 10), facecolor=dark)
    gs   = gridspec.GridSpec(1, 4, figure=fig,
                             width_ratios=[3.8, 0.03, 0.7, 0.03],
                             wspace=0.025)
    ax_p  = fig.add_subplot(gs[0])
    ax_c  = fig.add_subplot(gs[1])
    ax_vp = fig.add_subplot(gs[2])
    ax_c2 = fig.add_subplot(gs[3])

    for ax in [ax_p, ax_c, ax_vp, ax_c2]:
        ax.set_facecolor(dark)
        for sp in ax.spines.values():
            sp.set_color("#21262D")

    x = np.arange(len(window))

    # OHLC candlesticks
    _draw_ohlc(ax_p, window)

    # ── Confluence heatmap bands ──────────────────────────────────────────────
    visible = [z for z in zones if y_lo <= z.price <= y_hi]

    if visible:
        max_score = max(z.score for z in visible)
        cmap      = plt.cm.YlOrRd

        for zone in visible:
            ns     = zone.score / max_score
            colour = cmap(0.15 + 0.85 * ns)
            alpha  = 0.10 + 0.52 * ns
            hw     = bin_size * 1.3
            ax_p.axhspan(zone.price - hw, zone.price + hw,
                         color=colour, alpha=alpha, linewidth=0)

        # Annotate high-score zones
        top_vis = [z for z in visible if z.score > 0.50 * max_score][:show_n]
        for z in top_vis:
            ns     = z.score / max_score
            colour = cmap(0.15 + 0.85 * ns)
            tfs    = ",".join(sorted(z.timeframes))
            fts    = "|".join(sorted(z.ftypes))
            ax_p.annotate(
                f"{z.score:.1f} [{tfs}]\n{fts}",
                xy=(x[-1], z.price),
                xytext=(10, 0), textcoords="offset points",
                color=colour, fontsize=5.5, va="center",
                fontfamily="monospace",
            )

    # VAH / VAL reference lines per timeframe (POC removed)
    for tf, vp in profiles.items():
        c = TF_COLORS[tf]
        ax_p.axhline(vp.vah, color=c, lw=0.7, ls="--", alpha=0.45, label=f"VAH/VAL {tf}")
        ax_p.axhline(vp.val, color=c, lw=0.7, ls="--", alpha=0.45)

    # ── Date x-axis ───────────────────────────────────────────────────────────
    n_bars = len(window)
    dates  = window["date"].values   # numpy datetime64 array

    # Pick ~8 evenly spaced tick positions
    n_ticks  = 8
    tick_idx = np.linspace(0, n_bars - 1, n_ticks, dtype=int)

    def fmt_date(idx):
        ts = pd.Timestamp(dates[int(idx)])
        return ts.strftime("%b %d\n%H:%M")

    ax_p.set_xticks(tick_idx)
    ax_p.set_xticklabels([fmt_date(i) for i in tick_idx],
                          fontsize=7, color="#8B949E")

    ax_p.set_xlim(0, n_bars - 1)
    ax_p.set_ylim(y_lo, y_hi)
    ax_p.tick_params(axis="y", colors="#8B949E", labelsize=8)
    ax_p.tick_params(axis="x", colors="#8B949E", labelsize=7, length=3)
    ax_p.set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)
    ax_p.legend(loc="upper left", fontsize=7, framealpha=0.25,
                labelcolor="white", facecolor="#161B22", edgecolor="#21262D")

    mode    = "Anchored" if anchored else "Rolling"
    win_tag = "random" if random_window else "latest"
    d_start = date_start.strftime("%Y-%m-%d %H:%M")
    d_end   = date_end.strftime("%Y-%m-%d %H:%M")
    ax_p.set_title(
        f"NQ Stacked VP  ·  {mode}  ·  {d_start}  →  {d_end}  ({win_tag})",
        color="#F0F6FC", fontsize=10, pad=10, fontweight="bold",
    )

    # Colorbar for bands
    sm1 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=ax_c)
    cb1.set_label("Confluence", color="#8B949E", fontsize=7)
    cb1.ax.tick_params(colors="#8B949E", labelsize=6)

    # ── Composite VP heatmap (right strip) ───────────────────────────────────
    tf_list  = list(profiles.keys())
    ref_bins = profiles[tf_list[0]].bins if tf_list else np.array([])
    nb       = len(ref_bins)

    # Stack normalised volume profiles side-by-side, then average
    hmap = np.zeros((nb, len(tf_list)))
    for j, tf in enumerate(tf_list):
        vp   = profiles[tf]
        col  = np.interp(ref_bins, vp.bins, vp.vols, left=0, right=0)
        mx   = col.max()
        hmap[:, j] = col / mx if mx > 0 else col

    composite = hmap.mean(axis=1)

    # Blend with normalised confluence scores
    cf_layer = np.zeros(nb)
    if visible:
        mx_s = max(z.score for z in visible)
        for z in visible:
            ci = int(np.argmin(np.abs(ref_bins - z.price)))
            cf_layer[ci] = max(cf_layer[ci], z.score / mx_s)

    final_hmap = 0.30 * composite + 0.70 * cf_layer

    extent = [0, 1,
              float(ref_bins[0])  if nb else 0,
              float(ref_bins[-1]) if nb else 1]
    ax_vp.imshow(final_hmap.reshape(-1, 1), aspect="auto", origin="lower",
                 cmap="YlOrRd", extent=extent, vmin=0, vmax=1)
    ax_vp.set_xlim(0, 1)
    ax_vp.set_ylim(y_lo, y_hi)
    ax_vp.set_xticks([])
    ax_vp.yaxis.set_label_position("right")
    ax_vp.yaxis.tick_right()
    ax_vp.tick_params(colors="#8B949E", labelsize=7)
    ax_vp.set_title("VP\nHeat", color="#F0F6FC", fontsize=8, pad=6)

    sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, cax=ax_c2)
    cb2.ax.tick_params(colors="#8B949E", labelsize=6)

    plt.tight_layout(pad=0.6)

    mode_tag = "anchored" if anchored else "rolling"
    win_slug  = date_start.strftime("%Y%m%d_%H%M")
    out_file  = f"stacked_vp_{mode_tag}_{win_slug}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=dark)
    print(f"\n  Saved → {out_file}")
    plt.show()


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    zones: list,
    tolerance: float,
    years: int = 3,
    lookback_bars: int = 30,
    forward_bars: int = 240,
    min_sep_bars: int = 60,
    reversal_pct: float = 0.0035,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    For each confluence zone, scan the past `years` years of 1-minute data and
    find every touch.  A touch is a bar whose [low, high] range overlaps the
    zone ± tolerance.  A reversal is a subsequent move of ≥ reversal_pct from
    the touch price in the direction opposite to the approach.

    Approach direction:
      - 'resistance': close lookback_bars before touch was above zone + tolerance
      - 'support':    close lookback_bars before touch was below zone - tolerance
      - 'ambiguous':  price was already inside the zone (accept any ≥ threshold move)
    """
    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df3 = df[df["date"] >= start_date].reset_index(drop=True)

    hi  = df3["high"].values.astype(np.float64)
    lo  = df3["low"].values.astype(np.float64)
    cl  = df3["close"].values.astype(np.float64)
    n   = len(df3)
    if not quiet:
        print(f"  Backtest window: {df3['date'].iloc[0].date()} → {df3['date'].iloc[-1].date()}  ({n:,} bars)")

    rows = []
    for zone in zones:
        p   = zone.price
        tol = tolerance

        # All bars where price overlaps the zone
        raw_touches = np.where((lo <= p + tol) & (hi >= p - tol))[0]

        # Deduplicate: keep first touch in any run of min_sep_bars
        deduped: list[int] = []
        last_t = -min_sep_bars
        for t in raw_touches:
            if t - last_t >= min_sep_bars:
                deduped.append(t)
                last_t = t

        total     = len(deduped)
        reversals = 0
        rev_moves: list[float] = []

        for t in deduped:
            touch_price = cl[t]
            threshold   = reversal_pct * touch_price

            # Approach direction
            prev_idx   = max(0, t - lookback_bars)
            prev_close = cl[prev_idx]
            if prev_close > p + tol:
                direction = "resistance"
            elif prev_close < p - tol:
                direction = "support"
            else:
                direction = "ambiguous"

            # Forward window
            fwd_end = min(n, t + forward_bars + 1)
            if fwd_end <= t:
                continue
            fwd_hi = hi[t:fwd_end].max()
            fwd_lo = lo[t:fwd_end].min()

            if direction == "resistance":
                # Expect price to fall away: down-move must exceed threshold
                move = touch_price - fwd_lo
            elif direction == "support":
                # Expect price to rise away: up-move must exceed threshold
                move = fwd_hi - touch_price
            else:
                # Ambiguous: accept either direction
                move = max(touch_price - fwd_lo, fwd_hi - touch_price)

            if move >= threshold:
                reversals += 1
                rev_moves.append(move / touch_price * 100)

        rows.append({
            "price":        round(zone.price, 2),
            "score":        round(zone.score, 1),
            "n_tf":         zone.n_tf,
            "timeframes":   ",".join(sorted(zone.timeframes)),
            "ftypes":       "|".join(sorted(zone.ftypes)),
            "touches":      total,
            "reversals":    reversals,
            "reversal_rate": round(reversals / total, 3) if total > 0 else 0.0,
            "avg_rev_pct":  round(float(np.mean(rev_moves)), 3) if rev_moves else 0.0,
        })

    return pd.DataFrame(rows).sort_values("reversal_rate", ascending=False)


def print_backtest_report(bt: pd.DataFrame, reversal_pct: float = 0.0035):
    pct_str = f"{reversal_pct*100:.2f}%"
    sep = "-" * 88
    print(f"\n{sep}")
    print(f"  BACKTEST — reversal threshold {pct_str} of price  (3-year window)")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>5}  {'TF':>2}  {'TOUCHES':>7}  "
          f"{'REVS':>5}  {'RATE':>6}  {'AVG_REV%':>8}  TYPES")
    print(sep)
    for _, r in bt.iterrows():
        flag = " *" if r["reversal_rate"] >= 0.60 else "  "
        print(f"  {r['price']:>9.2f}  {r['score']:>5.1f}  {r['n_tf']:>2}  "
              f"{r['touches']:>7}  {r['reversals']:>5}  "
              f"{r['reversal_rate']:>5.1%}  {r['avg_rev_pct']:>7.3f}%  "
              f"{r['ftypes']}{flag}")
    print(sep)
    above = bt[bt["reversal_rate"] >= 0.50]
    elite = bt[bt["reversal_rate"] >= 0.60]
    print(f"  Zones >= 50% reversal rate : {len(above)}")
    print(f"  Zones >= 60% reversal rate : {len(elite)}  (* in table)")
    if len(above):
        print(f"  Avg score of >=50% zones   : {above['score'].mean():.1f}  "
              f"(all zones avg: {bt['score'].mean():.1f})")


# ─── Gradient Tuning Sweep ────────────────────────────────────────────────────

def tune_shelf_gradient(
    df: pd.DataFrame,
    profiles: dict,
    bin_size: float,
    reversal_pct: float = 0.0035,
    shelf_steps: int = 20,
) -> pd.DataFrame:
    """
    Sweep SHELF_SLOPE_MAX from 0.01 → 0.30 while holding LEDGE_SLOPE_MIN fixed.
    For each value: rebuild features, restack zones, run the full backtest.
    Reports touch-weighted reversal rate and zone counts so we can find the
    inflection point where tighter/looser shelf classification helps most.
    """
    shelf_vals = np.round(np.linspace(0.01, 0.30, shelf_steps), 4)

    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  SHELF_SLOPE_MAX sweep  (LEDGE_SLOPE_MIN fixed at {LEDGE_SLOPE_MIN})")
    print(f"  reversal threshold: {reversal_pct*100:.2f}%  |  {shelf_steps} steps")
    print(sep)
    print(f"  {'SHELF_MAX':>9}  {'SHELVES':>7}  {'ZONES':>5}  "
          f"{'W_RATE':>7}  {'AVG_RATE':>8}  {'MAX_RATE':>8}  {'>=50%':>5}  {'>=60%':>5}")
    print(sep)

    rows = []
    for sh in shelf_vals:
        # Re-detect features with this shelf threshold, LVN/VAH/VAL unchanged
        all_features = []
        for vp in profiles.values():
            all_features.extend(detect_features(vp, shelf_max=sh))
        zones = stack_features(all_features, bin_size)

        n_shelf = sum(1 for f in all_features if f.ftype == "shelf")

        bt = backtest(df, zones, tolerance=bin_size,
                      reversal_pct=reversal_pct, quiet=True)

        if len(bt) == 0 or bt["touches"].sum() == 0:
            w_rate = avg_rate = max_rate = n50 = n60 = 0.0
        else:
            total_t = bt["touches"].sum()
            w_rate  = float((bt["reversal_rate"] * bt["touches"]).sum() / total_t)
            avg_rate = float(bt["reversal_rate"].mean())
            max_rate = float(bt["reversal_rate"].max())
            n50 = int((bt["reversal_rate"] >= 0.50).sum())
            n60 = int((bt["reversal_rate"] >= 0.60).sum())

        print(f"  {sh:>9.4f}  {n_shelf:>7}  {len(zones):>5}  "
              f"{w_rate:>7.1%}  {avg_rate:>8.1%}  {max_rate:>8.1%}  {n50:>5}  {n60:>5}")

        rows.append({
            "shelf_max":  sh,
            "n_shelf":    n_shelf,
            "n_zones":    len(zones),
            "w_rate":     round(w_rate,   4),
            "avg_rate":   round(avg_rate, 4),
            "max_rate":   round(max_rate, 4),
            "n_50pct":    n50,
            "n_60pct":    n60,
        })

    print(sep)
    result = pd.DataFrame(rows)

    # Highlight the best row by weighted rate
    best = result.loc[result["w_rate"].idxmax()]
    print(f"\n  Best shelf_max by touch-weighted rate: {best['shelf_max']:.4f}  "
          f"(w_rate={best['w_rate']:.1%}, >=50%: {int(best['n_50pct'])})")

    out = "gradient_sweep.csv"
    result.to_csv(out, index=False)
    print(f"  Full sweep saved → {out}")
    return result


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    global DISPLAY_BARS
    parser = argparse.ArgumentParser(
        description="NQ Stacked Volume Profile — confluence zone detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--anchored", action="store_true",
                        help="Calendar-anchored profiles (default: rolling)")
    parser.add_argument("--compare",  action="store_true",
                        help="Run both rolling and anchored, compare results")
    parser.add_argument("--bins",     type=float, default=BIN_SIZE,
                        help=f"Bin size in NQ points (default {BIN_SIZE})")
    parser.add_argument("--display",  type=int,   default=DISPLAY_BARS,
                        help=f"Recent 1-min bars in chart (default {DISPLAY_BARS})")
    parser.add_argument("--top",      type=int,   default=20,
                        help="Top N zones in console report (default 20)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run 3-year reversal backtest on detected zones")
    parser.add_argument("--rev-pct",  type=float, default=0.35,
                        help="Reversal threshold in %% of price (default 0.35)")
    parser.add_argument("--random",         action="store_true",
                        help="Show a random historical window instead of latest bars")
    parser.add_argument("--seed",           type=int,   default=None,
                        help="Random seed for --random (for reproducibility)")
    parser.add_argument("--tune-gradients", action="store_true",
                        help="Sweep SHELF_SLOPE_MAX and report backtest metrics per step")
    args = parser.parse_args()

    DISPLAY_BARS = args.display
    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 52)
    print("  NQ STACKED VOLUME PROFILE ANALYZER")
    print("=" * 52)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df):,} bars  |  {df['date'].min().date()} → {df['date'].max().date()}")

    modes = [False, True] if args.compare else [args.anchored]

    results = {}
    for anchored in modes:
        label = "ANCHORED" if anchored else "ROLLING"
        print(f"\n{'-'*52}")
        print(f"  {label} profiles")
        print(f"{'-'*52}")
        zones, profiles = run(df, anchored=anchored, bin_size=args.bins)
        print_report(zones, show_n=args.top)
        results[label] = zones

        if args.tune_gradients:
            tune_shelf_gradient(df, profiles, bin_size=args.bins,
                                reversal_pct=args.rev_pct / 100.0)

        if args.backtest:
            rev_threshold = args.rev_pct / 100.0
            print(f"\nRunning backtest (threshold={args.rev_pct}%)...")
            bt = backtest(
                df, zones,
                tolerance=args.bins,
                reversal_pct=rev_threshold,
            )
            print_backtest_report(bt, reversal_pct=rev_threshold)
            bt_file = f"backtest_{'anchored' if anchored else 'rolling'}.csv"
            bt.to_csv(bt_file, index=False)
            print(f"  Full results saved → {bt_file}")

        plot(df, zones, profiles, anchored=anchored,
             bin_size=args.bins, show_n=args.top,
             random_window=args.random)

    if args.compare and len(results) == 2:
        print("\n  -- COMPARISON ------------------------------------------")
        for label, zones in results.items():
            high = sum(1 for z in zones if z.n_tf >= 3)
            top4 = sum(1 for z in zones if z.n_tf == 4)
            print(f"  {label:<10}  3+tf={high:>3}  4tf={top4:>3}")


if __name__ == "__main__":
    main()
