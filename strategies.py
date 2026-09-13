"""Strategy layer: each strategy maps market context -> daily target weight.

Weights are in [0, max_weight] on the LETF; the uninvested remainder sits in
cash (earns the risk-free rate in the backtest). All inputs are causal, so the
weight for day t uses only information known at the close of t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EngineConfig


def w_buy_hold(ctx: dict, cfg: EngineConfig) -> pd.Series:
    idx = ctx["letf_ret"].index
    return pd.Series(1.0, index=idx, name="buy_hold")


def w_regime_sma(ctx: dict, cfg: EngineConfig) -> pd.Series:
    reg = ctx["regime_sma"].reindex(ctx["letf_ret"].index).fillna(0.0)
    return reg.clip(0.0, 1.0).rename("regime_sma")


def w_vol_target(ctx: dict, cfg: EngineConfig) -> pd.Series:
    vol = ctx["garch_vol"].reindex(ctx["letf_ret"].index)
    w = (cfg.target_vol / vol).clip(0.0, cfg.max_weight)
    return w.fillna(0.0).rename("vol_target")


def w_vol_target_regime(ctx: dict, cfg: EngineConfig) -> pd.Series:
    return (w_vol_target(ctx, cfg) * w_regime_sma(ctx, cfg)).rename("vol_target_regime")


def w_vol_target_hmm(ctx: dict, cfg: EngineConfig) -> pd.Series:
    reg = ctx["regime_hmm"].reindex(ctx["letf_ret"].index).fillna(1.0)
    return (w_vol_target(ctx, cfg) * reg).rename("vol_target_hmm")


def w_vol_target_har(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Vol-targeting that sizes off the HAR-RV forecast instead of GARCH."""
    vol = ctx["har_vol"].reindex(ctx["letf_ret"].index)
    w = (cfg.target_vol / vol).clip(0.0, cfg.max_weight)
    return w.fillna(0.0).rename("vol_target_har")


def w_regime_tsmom(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Long/flat on 12-month time-series momentum of the underlying."""
    sig = ctx["tsmom"].reindex(ctx["letf_ret"].index).fillna(0.0)
    return sig.clip(0.0, 1.0).rename("regime_tsmom")


def w_vol_target_tsmom(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """HAR/GARCH vol-target gated by time-series-momentum trend."""
    return (w_vol_target(ctx, cfg) * w_regime_tsmom(ctx, cfg)).rename("vol_target_tsmom")


def w_regime_autocorr(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Compounding-aware tilt from return autocorrelation (arXiv 2504.20116).

    Positive AR(1) (trending → favorable LETF compounding) raises exposure;
    negative AR(1) (mean-reverting → decay dominates) cuts it. Centered at 0.5.
    """
    ac = ctx["ar1"].reindex(ctx["letf_ret"].index).fillna(0.0)
    tilt = (0.5 + cfg.autocorr_gain * ac).clip(0.0, 1.0)
    return tilt.rename("regime_autocorr")


def w_vol_target_autocorr(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Vol-target sized further by the autocorrelation (LETF-dynamics) tilt."""
    return (w_vol_target(ctx, cfg) * w_regime_autocorr(ctx, cfg)).rename("vol_target_autocorr")


def w_vol_target_armod(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """HAR vol-target whose TARGET is *modulated* by return autocorrelation.

    The fix for vol_target_autocorr's over-de-risking: AR(1) scales the vol
    target (trending → higher target → more exposure; mean-reverting → lower)
    rather than multiplying the final weight, so it tilts without stacking two
    independent risk cuts. Sizes off HAR-RV (the better vol forecaster).
    """
    vol = ctx["har_vol"].reindex(ctx["letf_ret"].index)
    ac = ctx["ar1"].reindex(ctx["letf_ret"].index).fillna(0.0)
    eff_target = (cfg.target_vol * (1.0 + cfg.autocorr_gain * ac)).clip(lower=0.0)
    w = (eff_target / vol).clip(0.0, cfg.max_weight)
    return w.fillna(0.0).rename("vol_target_armod")


def _apply_lowturn(w: pd.Series, every: int, band: float, name: str) -> pd.Series:
    """Collapse daily vol-target churn into a few trades/year (T+1 cash-safe):
    revisit the target only every `every` days, and only move the held weight
    when the new target differs from it by more than `band`."""
    vals = w.fillna(0.0).values
    held = np.empty(len(vals))
    cur = 0.0
    for i in range(len(vals)):
        if i % every == 0 and abs(vals[i] - cur) > band:
            cur = float(vals[i])
        held[i] = cur
    return pd.Series(held, index=w.index, name=name)


def w_vol_target_har_live(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Deployable low-turnover vol_target_har: weekly revisit + no-trade band.

    Same validated HAR vol-target signal, but only acted on weekly and only when
    the target moves materially — turning ~5x daily turnover into a handful of
    trades a year so it actually runs on a T+1 cash account.
    """
    base = w_vol_target_har(ctx, cfg)
    return _apply_lowturn(base, cfg.live_rebalance_days, cfg.live_band,
                          "vol_target_har_live")


def w_vol_target_gjr(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Vol-target sized off the GJR-GARCH-t forecast (asymmetric leverage effect).

    Cuts leverage one bar earlier into selloffs than symmetric GARCH — the panel
    rank-3 idea, aimed at the grind-style drawdowns (2000/2008) that drove the
    naked-3x wipeout, not the 1-2 day event gaps the IV floor targeted.
    """
    vol = ctx["gjr_vol"].reindex(ctx["letf_ret"].index)
    w = (cfg.target_vol / vol).clip(0.0, cfg.max_weight)
    return w.fillna(0.0).rename("vol_target_gjr")


def w_vol_target_ivhar(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Vol-target sized off the IV-floored HAR vol (multi-agent panel rank 2).

    Same HAR vol-targeting as vol_target_har, but the vol denominator is floored
    by a fraction of forward implied vol, lifting the forecast (and cutting
    leverage) into scheduled-event spikes HAR can't see coming.
    """
    vol = ctx["iv_blended_vol"].reindex(ctx["letf_ret"].index)
    w = (cfg.target_vol / vol).clip(0.0, cfg.max_weight)
    return w.fillna(0.0).rename("vol_target_ivhar")



def _dd_throttle(base_w: pd.Series, letf_ret: pd.Series, edges: tuple,
                 mults: tuple, buffer_: float) -> pd.Series:
    """Hysteretic exposure multiplier keyed on the BASE rule's own drawdown.

    CAUSALITY: the equity proxy is SHIFTED ONE DAY before cummax/drawdown, so
    day t's multiplier uses only information through t-1 (the look-ahead trap
    the review panel flagged). Tiers step one level per day with a recovery
    buffer, and the proxy tracks the UN-throttled base rule so the 0.0 tier
    is never absorbing.
    """
    base_ret = (base_w * letf_ret).fillna(0.0)
    eq = (1.0 + base_ret).cumprod().shift(1)          # info through t-1 only
    dd = (eq / eq.cummax() - 1.0).fillna(0.0)
    tier = 0
    out = np.empty(len(dd))
    for i, d in enumerate(dd.values):
        if tier < len(edges) and d <= edges[tier]:
            tier += 1                                  # deteriorate one step
        elif tier > 0 and d >= edges[tier - 1] + buffer_:
            tier -= 1                                  # recover one step
        out[i] = mults[tier]
    return pd.Series(out, index=dd.index, name="dd_throttle")


def w_vol_target_har_live_dd(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Agenda #1: the live low-turnover rule with a hysteretic drawdown throttle."""
    base = w_vol_target_har_live(ctx, cfg)
    mult = _dd_throttle(base, ctx["letf_ret"], cfg.dd_edges, cfg.dd_mults, cfg.dd_buffer)
    return (base * mult).rename("vol_target_har_live_dd")


def w_vol_target_har_live_vix(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Agenda #2: live low-turnover rule x VIX term-structure stress multiplier.

    The gate multiplies the HELD (post-lowturn) weight daily, so it only adds
    turnover during backwardation stress — 1.0 the rest of the time. Neutral
    before VIX3M history exists (~2007), so the pre-2007 sample matches the base.
    """
    base = w_vol_target_har_live(ctx, cfg)
    gate = ctx["vix_gate"].reindex(base.index).fillna(1.0)
    return (base * gate).rename("vol_target_har_live_vix")


def w_vol_target_har_live_kelly(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Agenda #4: fractional-Kelly CEILING on the daily HAR target, then the
    same weekly/8%-band low-turnover wrapper as the live rule. Never raises
    exposure above the vol target — only caps it when slow drift is weak."""
    daily = w_vol_target_har(ctx, cfg)
    kelly = ctx["kelly_w"].reindex(daily.index).fillna(cfg.max_weight)
    capped = pd.concat([daily, kelly], axis=1).min(axis=1)
    return _apply_lowturn(capped, cfg.live_rebalance_days, cfg.live_band,
                          "vol_target_har_live_kelly")


def w_vol_target_har_live_cvar(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Agenda #5: CVaR throttle x daily HAR target, then the low-turnover
    wrapper. The CVaR budget is derived from target_vol (Gaussian 5% ES), so the
    throttle only binds when the realized tail is fatter than the target allows."""
    daily = w_vol_target_har(ctx, cfg)
    scale = ctx["cvar_scale"].reindex(daily.index).fillna(1.0)
    return _apply_lowturn(daily * scale, cfg.live_rebalance_days, cfg.live_band,
                          "vol_target_har_live_cvar")


def w_tom(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """Pure turn-of-month anomaly: fully invested on the TOM window (last
    trading day through the first three of each month), cash otherwise.
    ~4/21 days exposed. The matched-null rotation test is the honest judge:
    it asks whether THESE calendar days beat random days at equal exposure."""
    flag = ctx["tom"].reindex(ctx["letf_ret"].index).fillna(0.0)
    return flag.rename("tom")



def w_ema_confluence_live(ctx: dict, cfg: EngineConfig) -> pd.Series:
    """YouTube-sourced test (LewisWJackson, weekly swing rule), applied to daily
    bars long-only: in the market only when EMA21 > EMA50 > EMA200, RSI(14) > 50
    and MACD(12,26,9) histogram > 0 on the underlying; cash otherwise. All
    parameters PRE-COMMITTED as stated in the video. Signal is SHIFTED one day
    (underlying_close is the raw same-day close) then wrapped in the same
    weekly/8%-band low-turnover rule as the live strategy.
    """
    px = ctx["underlying_close"].astype(float)
    e21, e50, e200 = (px.ewm(span=n, adjust=False).mean() for n in (21, 50, 200))
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    on = ((e21 > e50) & (e50 > e200) & (rsi > 50) & (hist > 0)).astype(float)
    on = on.shift(1).reindex(ctx["letf_ret"].index).fillna(0.0)      # causal
    return _apply_lowturn(on, cfg.live_rebalance_days, cfg.live_band,
                          "ema_confluence_live")


REGISTRY = {
    "buy_hold": w_buy_hold,
    "regime_sma": w_regime_sma,
    "vol_target": w_vol_target,
    "vol_target_regime": w_vol_target_regime,
    "vol_target_hmm": w_vol_target_hmm,
    "vol_target_har": w_vol_target_har,
    "regime_tsmom": w_regime_tsmom,
    "vol_target_tsmom": w_vol_target_tsmom,
    "regime_autocorr": w_regime_autocorr,
    "vol_target_autocorr": w_vol_target_autocorr,
    "vol_target_armod": w_vol_target_armod,
    "vol_target_ivhar": w_vol_target_ivhar,
    "vol_target_gjr": w_vol_target_gjr,
    "vol_target_har_live": w_vol_target_har_live,
    "vol_target_har_live_dd": w_vol_target_har_live_dd,
    "vol_target_har_live_vix": w_vol_target_har_live_vix,
    "vol_target_har_live_kelly": w_vol_target_har_live_kelly,
    "vol_target_har_live_cvar": w_vol_target_har_live_cvar,
    "tom": w_tom,
    "ema_confluence_live": w_ema_confluence_live,
}


def build_weights(name: str, ctx: dict, cfg: EngineConfig) -> pd.Series:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](ctx, cfg)
