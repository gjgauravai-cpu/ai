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
}


def build_weights(name: str, ctx: dict, cfg: EngineConfig) -> pd.Series:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](ctx, cfg)
