"""intraday15.py -- Can a 15m/1h-decided intraday strategy beat the existing
daily vol-target strategy on TQQQ, net of costs?

Strategies 1-7 are long-only, AT MOST ONE round trip per calendar day (the
cash-account / T+1 -safe design of the live engine). Strategies 8-9 (mom_multi,
mr_multi) were added by an explicit addendum confirming intraday trading is
permitted for this test: they allow up to 5 round trips/day and are NOT
cash-account safe -- they are a separate research question, reported in the
same tables, never used to override the T+1 constraint on strategies 1-7.

All decisions are strictly causal (act only on completed bars, enter/exit at
the close of the decision bar). Costs are `bps_per_crossing` on both buy and
sell (a round trip pays 2x). Parameters are pre-committed -- nothing here is
tuned against the results.

Run: `python intraday15.py` from this directory.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from datetime import time as dtime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TICKER = "TQQQ"
ET = "America/New_York"
BASE_COST_BPS = 3.0       # one-side cost per crossing; matches config.py TQQQ half_spread_bps
STRESS_COST_BPS = 5.0
N_PERM = 500
SEED = 7
TRADING_DAYS = 252
MAX_RT_MULTI = 5           # strategies 8-9 only (addendum)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def fetch_intraday(interval: str, period: str) -> pd.DataFrame:
    """Regular-hours (09:30-16:00 ET) OHLCV bars for TICKER."""
    import yfinance as yf

    raw = yf.download(TICKER, interval=interval, period=period, auto_adjust=True,
                       prepost=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no {interval}/{period} data for {TICKER}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df = df.between_time("09:30", "15:59:59").dropna()
    df.index.name = "datetime"
    return df


class Session(NamedTuple):
    """Cleaned bars split into full (modal bar-count) trading days for analysis."""
    by_day: dict           # date -> DataFrame (OHLCV, datetime index)
    day_list: list         # sorted dates with the modal bar count
    bars_per_day: int
    close_matrix: np.ndarray   # shape (n_days, bars_per_day)
    day_index: pd.DatetimeIndex
    analysis_bars: int
    dropped_partial_days: int


def build_session(df: pd.DataFrame) -> Session:
    dates = df.index.normalize()
    groups = {d: g for d, g in df.groupby(dates)}
    counts = pd.Series({d: len(g) for d, g in groups.items()})
    modal = int(counts.mode().iloc[0])
    by_day = {d: g for d, g in groups.items() if len(g) == modal}
    day_list = sorted(by_day.keys())
    close_matrix = np.array([by_day[d]["close"].to_numpy() for d in day_list])
    return Session(
        by_day=by_day, day_list=day_list, bars_per_day=modal,
        close_matrix=close_matrix, day_index=pd.DatetimeIndex(day_list),
        analysis_bars=modal * len(day_list),
        dropped_partial_days=len(groups) - len(by_day),
    )


# --------------------------------------------------------------------------- #
# Trades + core numeric helpers                                              #
# --------------------------------------------------------------------------- #
class Trade(NamedTuple):
    day_idx: int
    entry_offset: int
    exit_offset: int
    cross_day: bool         # True: entry at day_idx's last bar, exit at (day_idx+1)'s exit_offset bar


def trade_gross(t: Trade, close_matrix: np.ndarray) -> float:
    ep = close_matrix[t.day_idx, t.entry_offset]
    xp = close_matrix[t.day_idx + 1, t.exit_offset] if t.cross_day else close_matrix[t.day_idx, t.exit_offset]
    return float(xp / ep - 1.0)


def daily_series(trades: list, close_matrix: np.ndarray, day_index: pd.DatetimeIndex,
                  cost_rt: float) -> pd.Series:
    s = np.zeros(len(day_index))
    for t in trades:
        net = trade_gross(t, close_matrix) - cost_rt
        pos = t.day_idx + 1 if t.cross_day else t.day_idx
        s[pos] += net
    return pd.Series(s, index=day_index)


def sharpe(daily: pd.Series) -> float:
    sd = daily.std(ddof=0)
    return float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd and sd > 0 else float("nan")


def max_dd_pct(daily: pd.Series) -> float:
    if len(daily) == 0:
        return float("nan")
    eq = (1.0 + daily).cumprod()
    return float((eq / eq.cummax() - 1.0).min()) * 100.0


# --------------------------------------------------------------------------- #
# Strategy trade generators (1 round trip / day, strategies 1-5)             #
# --------------------------------------------------------------------------- #
def gen_orb(day_list: list, by_day: dict) -> list:
    trades = []
    for i, d in enumerate(day_list):
        c = by_day[d]["close"].to_numpy()
        or_high = float(by_day[d]["high"].iloc[0])
        or_low = float(by_day[d]["low"].iloc[0])
        n = len(c)
        entry = next((j for j in range(1, n) if c[j] > or_high), None)
        if entry is None:
            continue
        exitb = next((m for m in range(entry + 1, n) if c[m] < or_low), n - 1)
        trades.append(Trade(i, entry, exitb, False))
    return trades


def gen_first_bar(day_list: list, by_day: dict, momentum: bool) -> list:
    trades = []
    for i, d in enumerate(day_list):
        b = by_day[d]
        n = len(b)
        r0 = float(b["close"].iloc[0]) / float(b["open"].iloc[0]) - 1.0
        if (r0 > 0) == momentum and r0 != 0:
            trades.append(Trade(i, 0, n - 1, False))
    return trades


def gen_vwap_reversion(day_list: list, by_day: dict) -> list:
    trades = []
    cutoff = dtime(11, 0)
    for i, d in enumerate(day_list):
        b = by_day[d]
        n = len(b)
        interval = b.index[1] - b.index[0] if n > 1 else pd.Timedelta(minutes=15)
        typical = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = b["volume"].astype(float)
        vol_safe = vol.where(vol > 0, 1.0)
        vwap = (typical * vol_safe).cumsum() / vol_safe.cumsum()
        close_times = (b.index + interval).time
        entry = next((j for j in range(n) if close_times[j] >= cutoff), None)
        if entry is None:
            continue
        if float(b["close"].iloc[entry]) < float(vwap.iloc[entry]):
            trades.append(Trade(i, entry, n - 1, False))
    return trades


def gen_overnight(day_list: list, by_day: dict, every_other: bool) -> list:
    step = 2 if every_other else 1
    return [Trade(i, len(by_day[day_list[i]]) - 1, 0, True)
            for i in range(0, len(day_list) - 1, step)]


# --------------------------------------------------------------------------- #
# Addendum strategies: up to MAX_RT_MULTI round trips/day, always flat EOD    #
# --------------------------------------------------------------------------- #
def gen_mom_multi(day_list: list, by_day: dict, max_rt: int = MAX_RT_MULTI) -> list:
    trades = []
    for i, d in enumerate(day_list):
        c = by_day[d]["close"].to_numpy()
        n = len(c)
        j, n_rt = 3, 0
        while j < n and n_rt < max_rt:
            if c[j] > c[j - 1] > c[j - 2] > c[j - 3]:
                entry = j
                exitb = next((m for m in range(j + 1, n) if c[m] < c[m - 1]), n - 1)
                trades.append(Trade(i, entry, exitb, False))
                n_rt += 1
                j = exitb + 1
            else:
                j += 1
    return trades


def gen_mr_multi(day_list: list, by_day: dict, max_rt: int = MAX_RT_MULTI,
                  window: int = 20, k: float = 1.0, hold: int = 2) -> list:
    """Rolling stats computed on the continuous (cross-day) bar series -- a
    live system would carry vol context across the open, not forget it each
    morning. This means the day's FIRST bar can trigger on an overnight gap;
    documented as a caveat rather than hidden."""
    all_close = np.concatenate([by_day[d]["close"].to_numpy() for d in day_list])
    r = np.full(len(all_close), np.nan)
    r[1:] = all_close[1:] / all_close[:-1] - 1.0
    roll_sd = pd.Series(r).rolling(window).std(ddof=0).shift(1).to_numpy()

    trades, offset = [], 0
    for i, d in enumerate(day_list):
        n = len(by_day[d])
        j, n_rt = 0, 0
        while j < n and n_rt < max_rt:
            g = offset + j
            if not np.isnan(roll_sd[g]) and not np.isnan(r[g]) and r[g] < -k * roll_sd[g]:
                exitb = min(j + hold, n - 1)
                trades.append(Trade(i, j, exitb, False))
                n_rt += 1
                j = exitb + 1
            else:
                j += 1
        offset += n
    return trades


STRATEGY_DEFS = {
    "orb": (gen_orb, False),
    "first_bar_momentum": (lambda dl, bd: gen_first_bar(dl, bd, True), False),
    "first_bar_reversal": (lambda dl, bd: gen_first_bar(dl, bd, False), False),
    "vwap_reversion": (gen_vwap_reversion, False),
    "overnight_daily": (lambda dl, bd: gen_overnight(dl, bd, False), True),
    "overnight_altday": (lambda dl, bd: gen_overnight(dl, bd, True), True),
    "mom_multi": (gen_mom_multi, False),
    "mr_multi": (gen_mr_multi, False),
}


# --------------------------------------------------------------------------- #
# Metrics + matched-null significance test                                   #
# --------------------------------------------------------------------------- #
def strategy_metrics(name: str, trades: list, sess: Session, cost_bps: float,
                      cross_day: bool) -> dict:
    cost_rt = 2 * cost_bps / 1e4
    n = len(trades)
    gross_list = [trade_gross(t, sess.close_matrix) for t in trades]
    net_list = [g - cost_rt for g in gross_list]
    exposure_bars = sum((t.exit_offset - t.entry_offset) for t in trades if not t.cross_day)
    gross_total = (float(np.prod([1 + g for g in gross_list])) - 1.0) * 100 if n else 0.0
    net_total = (float(np.prod([1 + r for r in net_list])) - 1.0) * 100 if n else 0.0
    daily = daily_series(trades, sess.close_matrix, sess.day_index, cost_rt)
    sh = sharpe(daily)
    on_days = daily.to_numpy() != 0.0
    hit = float((daily.to_numpy()[on_days] > 0).mean()) * 100 if on_days.any() else float("nan")
    n_days = len(sess.day_list)
    if cross_day:
        exposure = n / max(n_days - 1, 1) * 100
    else:
        exposure = exposure_bars / sess.analysis_bars * 100 if sess.analysis_bars else float("nan")
    return {
        "strategy": name, "round_trips": n, "trades_per_day": n / n_days if n_days else float("nan"),
        "exposure_pct": exposure, "gross_total_pct": gross_total, "net_total_pct": net_total,
        "cost_drag_pct": gross_total - net_total, "sharpe_net": sh, "hit_rate_pct": hit,
        "max_dd_pct": max_dd_pct(daily), "avg_pnl_bps": float(np.mean(net_list) * 1e4) if n else float("nan"),
        "breakeven_bps": 2 * cost_bps, "daily": daily,
    }


def matched_null_pvalue(trades: list, sess: Session, cost_bps: float, real_sharpe: float,
                         cross_day: bool) -> str:
    """Keep each trading day's realized (entry_offset, exit_offset) bundle fixed;
    randomly permute WHICH days get which bundle (500 draws, default_rng(7)).
    p = fraction of permuted net Sharpes >= the real net Sharpe."""
    if not trades or np.isnan(real_sharpe):
        return "n/a"
    cost_rt = 2 * cost_bps / 1e4
    pool_n = (len(sess.day_list) - 1) if cross_day else len(sess.day_list)
    bundles: dict = {}
    for t in trades:
        bundles.setdefault(t.day_idx, []).append((t.entry_offset, t.exit_offset))
    bundle_list = list(bundles.values())
    k = len(bundle_list)
    if k == 0 or pool_n <= 0:
        return "n/a"
    if k >= pool_n:
        return "n/a (trades every eligible day)"

    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_PERM):
        days = rng.choice(pool_n, size=k, replace=False)
        order = rng.permutation(k)
        perm_trades = [Trade(int(days[slot]), eo, xo, cross_day)
                        for slot, bidx in enumerate(order)
                        for (eo, xo) in bundle_list[bidx]]
        sh = sharpe(daily_series(perm_trades, sess.close_matrix, sess.day_index, cost_rt))
        if not np.isnan(sh) and sh >= real_sharpe:
            hits += 1
    return f"{hits / N_PERM:.3f}"


# --------------------------------------------------------------------------- #
# Benchmarks: buy_hold + the live daily_vol_target strategy                  #
# --------------------------------------------------------------------------- #
def buy_hold_metrics(sess: Session, cost_bps: float) -> dict:
    closes = pd.Series(sess.close_matrix[:, -1], index=sess.day_index)
    daily_ret = closes.pct_change().dropna()
    gross_total = (float(sess.close_matrix[-1, -1]) / float(sess.close_matrix[0, 0]) - 1.0) * 100
    net_total = ((1 + gross_total / 100) * (1 - 2 * cost_bps / 1e4) - 1.0) * 100
    n_days = len(sess.day_list)
    return {
        "strategy": "buy_hold", "round_trips": 1, "trades_per_day": 1.0 / n_days if n_days else float("nan"),
        "exposure_pct": 100.0, "gross_total_pct": gross_total, "net_total_pct": net_total,
        "cost_drag_pct": gross_total - net_total, "sharpe_net": sharpe(daily_ret),
        "hit_rate_pct": float((daily_ret > 0).mean()) * 100 if len(daily_ret) else float("nan"),
        "max_dd_pct": max_dd_pct(daily_ret), "avg_pnl_bps": net_total * 100,
        "breakeven_bps": 2 * cost_bps, "daily": daily_ret,
    }


def load_daily_vol_target() -> dict | None:
    """Build the live vol_target_har_live strategy's daily net/gross/weight
    series (full history) at both cost levels, per the prescribed recipe."""
    try:
        import data
        import models
        from backtest import run_backtest
        from config import UNIVERSE, EngineConfig
        from strategies import build_weights

        cfg = EngineConfig()
        df = data.load_one("TQQQ", "2010-01-01", None, refresh=True)
        ret = data.daily_returns(df).dropna()
        ctx = {
            "letf_ret": ret, "letf_df": df,
            "underlying_close": data.load_one("QQQ", "2010-01-01", None)["close"],
            "har_vol": models.har_vol(df, cfg.garch_refit_every, cfg.garch_min_obs, cfg.trading_days),
        }
        w = build_weights("vol_target_har_live", ctx, cfg)
        out = {}
        for cost_bps in (BASE_COST_BPS, STRESS_COST_BPS):
            spec = replace(UNIVERSE[TICKER], half_spread_bps=cost_bps)
            bt = run_backtest(ret, w, spec, cfg, cfg.fallback_rf)
            out[cost_bps] = {"gross": bt["gross"], "net": bt["net"], "weight": bt["weight"]}
        return out
    except Exception as exc:                                    # noqa: BLE001
        print(f"[daily_vol_target] WARNING benchmark unavailable: {exc}")
        return None


def daily_vol_target_metrics(dvt: dict, cost_bps: float, plain_dates: pd.DatetimeIndex) -> dict:
    bt = dvt[cost_bps]
    net = bt["net"].reindex(plain_dates).dropna()
    gross = bt["gross"].reindex(plain_dates).dropna()
    w = bt["weight"].reindex(net.index).fillna(0.0)
    n_days = len(net)
    gross_total = (float((1 + gross).prod()) - 1.0) * 100 if n_days else float("nan")
    net_total = (float((1 + net).prod()) - 1.0) * 100 if n_days else float("nan")
    invested = net[w > 1e-6]
    turns = int((w.diff().abs() > 1e-6).sum())
    return {
        "strategy": "daily_vol_target", "round_trips": turns,
        "trades_per_day": turns / n_days if n_days else float("nan"),
        "exposure_pct": float(w.mean() * 100) if n_days else float("nan"),
        "gross_total_pct": gross_total, "net_total_pct": net_total,
        "cost_drag_pct": gross_total - net_total, "sharpe_net": sharpe(net),
        "hit_rate_pct": float((invested > 0).mean()) * 100 if len(invested) else float("nan"),
        "max_dd_pct": max_dd_pct(net),
        "avg_pnl_bps": float(invested.mean() * 1e4) if len(invested) else float("nan"),
        "breakeven_bps": 2 * cost_bps, "daily": net,
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                              #
# --------------------------------------------------------------------------- #
COLS = ["round_trips", "trades_per_day", "exposure_pct", "gross_total_pct", "net_total_pct",
        "cost_drag_pct", "sharpe_net", "hit_rate_pct", "max_dd_pct", "avg_pnl_bps",
        "breakeven_bps", "p_value"]


def run_dataset(label: str, interval: str, period: str, dvt: dict | None) -> pd.DataFrame | None:
    print(f"\n{'=' * 110}\nDATASET: {label}  (interval={interval}, period={period})\n{'=' * 110}")
    try:
        raw = fetch_intraday(interval, period)
    except Exception as exc:                                    # noqa: BLE001
        print(f"[ERROR] could not fetch {label}: {exc}")
        return None
    sess = build_session(raw)
    print(f"Raw RTH bars: {len(raw)}  |  full trading days used: {len(sess.day_list)} "
          f"({sess.bars_per_day} bars/day; {sess.dropped_partial_days} partial days dropped)  |  "
          f"range: {sess.day_list[0].date()} -> {sess.day_list[-1].date()}")

    rows, stress_lines = [], []
    for name, (gen, cross_day) in STRATEGY_DEFS.items():
        trades = gen(sess.day_list, sess.by_day)
        m3 = strategy_metrics(name, trades, sess, BASE_COST_BPS, cross_day)
        m5 = strategy_metrics(name, trades, sess, STRESS_COST_BPS, cross_day)
        pv3 = matched_null_pvalue(trades, sess, BASE_COST_BPS, m3["sharpe_net"], cross_day)
        pv5 = matched_null_pvalue(trades, sess, STRESS_COST_BPS, m5["sharpe_net"], cross_day)
        row = {k: v for k, v in m3.items() if k != "daily"}
        row["p_value"] = pv3
        rows.append(row)
        stress_lines.append((name, m3["sharpe_net"], m5["sharpe_net"], pv3, pv5))

    bh3 = buy_hold_metrics(sess, BASE_COST_BPS)
    bh5 = buy_hold_metrics(sess, STRESS_COST_BPS)
    rows.append({**{k: v for k, v in bh3.items() if k != "daily"}, "p_value": "n/a"})
    stress_lines.append(("buy_hold", bh3["sharpe_net"], bh5["sharpe_net"], "n/a", "n/a"))

    dvt_sh3 = dvt_sh5 = float("nan")
    if dvt is not None:
        plain_dates = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in sess.day_list])
        d3 = daily_vol_target_metrics(dvt, BASE_COST_BPS, plain_dates)
        d5 = daily_vol_target_metrics(dvt, STRESS_COST_BPS, plain_dates)
        rows.append({**{k: v for k, v in d3.items() if k != "daily"}, "p_value": "n/a (benchmark)"})
        stress_lines.append(("daily_vol_target", d3["sharpe_net"], d5["sharpe_net"], "n/a", "n/a"))
        dvt_sh3, dvt_sh5 = d3["sharpe_net"], d5["sharpe_net"]
    else:
        print("[daily_vol_target] SKIPPED (benchmark unavailable) -- see warning above.")

    table = pd.DataFrame(rows).set_index("strategy")[COLS]
    disp = table.copy()
    disp["round_trips"] = disp["round_trips"].astype(float)
    print("\n" + disp.to_string(float_format=lambda v: f"{v:,.2f}"))

    print(f"\nCOST-STRESS (5bps per crossing):")
    for name, s3, s5, p3, p5 in stress_lines:
        print(f"  {name:20s} sharpe@3bps={s3:6.2f}  sharpe@5bps={s5:6.2f}  "
              f"p@3bps={p3:>10}  p@5bps={p5:>10}")

    table.attrs["dvt_sharpe"] = {BASE_COST_BPS: dvt_sh3, STRESS_COST_BPS: dvt_sh5}
    table.attrs["sharpe_5bps"] = {r[0]: r[2] for r in stress_lines}
    table.attrs["p_5bps"] = {r[0]: r[4] for r in stress_lines}
    return table


def build_verdict(tables: dict) -> list:
    lines = []
    winners_by_dataset = {}
    for label, t in tables.items():
        if t is None:
            continue
        dvt_sh = t.attrs.get("dvt_sharpe", {})
        sh5 = t.attrs.get("sharpe_5bps", {})
        p5 = t.attrs.get("p_5bps", {})
        winners = []
        for strat, row in t.iterrows():
            if strat in ("buy_hold", "daily_vol_target"):
                continue
            p3 = row["p_value"]
            try:
                p3f = float(p3)
            except (TypeError, ValueError):
                continue
            p5v = p5.get(strat)
            try:
                p5f = float(p5v)
            except (TypeError, ValueError):
                continue
            beats3 = row["sharpe_net"] > dvt_sh.get(BASE_COST_BPS, float("nan"))
            beats5 = sh5.get(strat, float("nan")) > dvt_sh.get(STRESS_COST_BPS, float("nan"))
            if beats3 and beats5 and p3f < 0.05 and p5f < 0.05:
                winners.append(strat)
        winners_by_dataset[label] = winners

    any_winner = any(winners_by_dataset.values())
    lines.append("VERDICT:")
    if not any_winner:
        lines.append("No intraday strategy (1-9) beat daily_vol_target's net Sharpe over its own window")
        lines.append("with a matched-null p-value < 0.05 at BOTH 3bps and 5bps -- the bar (real edge +")
        lines.append("surviving realistic costs) was not cleared. Any apparent outperformance in the")
        lines.append("tables above is consistent with noise once costs and day-selection luck are priced in.")
    else:
        for label, winners in winners_by_dataset.items():
            if winners:
                lines.append(f"{label}: {', '.join(winners)} beat daily_vol_target's net Sharpe (same window)")
                lines.append("  with p < 0.05 at both 3bps and 5bps -- treat as a real, cost-surviving signal,")
                lines.append("  not proof of future performance, and re-verify out-of-sample before sizing it.")
        lines.append("All other strategies failed the bar above and should be treated as unproven.")
    return lines[:5] if len(lines) > 5 else lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dvt = load_daily_vol_target()

    tables = {
        "A: TQQQ 15m, 60d (yfinance max)": run_dataset("A: TQQQ 15m, 60d (yfinance max)", "15m", "60d", dvt),
        "B: TQQQ 1h, 730d (yfinance max)": run_dataset("B: TQQQ 1h, 730d (yfinance max)", "1h", "730d", dvt),
    }

    ok = {k: v for k, v in tables.items() if v is not None}
    if not ok:
        print("\n[FATAL] both datasets failed to load -- nothing to report.")
        sys.exit(1)
    if len(ok) < len(tables):
        missing = [k for k, v in tables.items() if v is None]
        print(f"\n[NOTE] dataset(s) unavailable, reporting only what succeeded: {missing}")

    frames = []
    for label, t in ok.items():
        f = t.reset_index()
        f.insert(0, "dataset", label)
        frames.append(f)
    combined = pd.concat(frames, ignore_index=True)
    csv_path = OUT_DIR / "intraday15_results.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved combined results table to {csv_path}")

    print("\n" + "=" * 110)
    for line in build_verdict(ok):
        print(line)
    print("=" * 110)


if __name__ == "__main__":
    main()
