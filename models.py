"""Time-series models: causal GARCH(1,1) vol forecasting + regime filters.

Everything here is strictly causal (walk-forward). A forecast for day t is built
only from information available at the close of day t-1, so the backtest has no
look-ahead bias.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Volatility models                                                           #
# --------------------------------------------------------------------------- #
def ewma_vol(returns: pd.Series, lam: float = 0.94,
             trading_days: int = 252) -> pd.Series:
    """RiskMetrics EWMA conditional vol (annualized), causal 1-step-ahead."""
    r = returns.dropna()
    var = np.empty(len(r))
    var[0] = r.iloc[:21].var(ddof=0) if len(r) > 21 else r.var(ddof=0)
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r.iloc[i - 1] ** 2
    return pd.Series(np.sqrt(var) * np.sqrt(trading_days), index=r.index,
                     name="ewma_vol")


def garch_vol(returns: pd.Series, refit_every: int = 21, min_obs: int = 252,
              trading_days: int = 252) -> pd.Series:
    """Walk-forward GARCH(1,1) 1-step-ahead conditional vol (annualized).

    Params are re-estimated every `refit_every` days on the expanding window;
    between refits the variance recursion is propagated with realized residuals.
    Falls back to EWMA on any estimation failure.
    """
    from arch import arch_model

    r = returns.dropna()
    r_pct = r * 100.0                      # arch is happiest with percent returns
    n = len(r)
    out = pd.Series(np.nan, index=r.index, name="garch_vol")
    if n <= min_obs:
        return ewma_vol(returns, trading_days=trading_days)

    omega = alpha = beta = mu = None
    sigma2 = None                          # variance forecast for the current day
    for pos in range(min_obs, n):
        refit = (pos - min_obs) % refit_every == 0 or omega is None
        if refit:
            try:
                res = arch_model(r_pct.iloc[:pos], mean="Constant",
                                 vol="GARCH", p=1, q=1, dist="normal").fit(
                    disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                beta = float(res.params["beta[1]"])
                mu = float(res.params["mu"])
                sigma2_prev = float(res.conditional_volatility.iloc[-1]) ** 2
            except Exception:              # noqa: BLE001 - degrade gracefully
                omega = None
                ew = ewma_vol(r.iloc[:pos + 1], trading_days=trading_days)
                out.iloc[pos] = ew.iloc[-1]
                continue
            resid_prev = r_pct.iloc[pos - 1] - mu
            sigma2 = omega + alpha * resid_prev ** 2 + beta * sigma2_prev
        else:
            resid_prev = r_pct.iloc[pos - 1] - mu
            sigma2 = omega + alpha * resid_prev ** 2 + beta * sigma2   # σ²_{t-1}=prev forecast
        out.iloc[pos] = np.sqrt(sigma2) / 100.0 * np.sqrt(trading_days)
    return out.ffill()


def gjr_vol(returns: pd.Series, refit_every: int = 21, min_obs: int = 252,
            trading_days: int = 252) -> pd.Series:
    """Walk-forward GJR-GARCH(1,1)-t 1-step-ahead conditional vol (annualized).

    Adds the leverage term γ·I(r<0)·r² to GARCH (negative shocks raise vol more
    than positive ones) plus Student-t innovations for the fat left tail, so the
    forecast rises one day EARLIER into a selloff — exactly when a 3x LETF must
    de-risk. Identical causal walk-forward scaffold and EWMA fallback as garch_vol.
    """
    from arch import arch_model

    r = returns.dropna()
    r_pct = r * 100.0
    n = len(r)
    out = pd.Series(np.nan, index=r.index, name="gjr_vol")
    if n <= min_obs:
        return ewma_vol(returns, trading_days=trading_days)

    omega = alpha = gamma = beta = mu = None
    sigma2 = None
    for pos in range(min_obs, n):
        refit = (pos - min_obs) % refit_every == 0 or omega is None
        if refit:
            try:
                res = arch_model(r_pct.iloc[:pos], mean="Constant",
                                 vol="GARCH", p=1, o=1, q=1, dist="t").fit(
                    disp="off", show_warning=False)
                omega = float(res.params["omega"])
                alpha = float(res.params["alpha[1]"])
                gamma = float(res.params["gamma[1]"])
                beta = float(res.params["beta[1]"])
                mu = float(res.params["mu"])
                sigma2_prev = float(res.conditional_volatility.iloc[-1]) ** 2
            except Exception:              # noqa: BLE001 - degrade gracefully
                omega = None
                ew = ewma_vol(r.iloc[:pos + 1], trading_days=trading_days)
                out.iloc[pos] = ew.iloc[-1]
                continue
            resid_prev = r_pct.iloc[pos - 1] - mu
            lev = gamma * resid_prev ** 2 * (resid_prev < 0)        # leverage term
            sigma2 = omega + alpha * resid_prev ** 2 + lev + beta * sigma2_prev
        else:
            resid_prev = r_pct.iloc[pos - 1] - mu
            lev = gamma * resid_prev ** 2 * (resid_prev < 0)
            sigma2 = omega + alpha * resid_prev ** 2 + lev + beta * sigma2
        out.iloc[pos] = np.sqrt(sigma2) / 100.0 * np.sqrt(trading_days)
    return out.ffill()


# --------------------------------------------------------------------------- #
# Regime models                                                               #
# --------------------------------------------------------------------------- #
def sma_regime(underlying_close: pd.Series, window: int = 200) -> pd.Series:
    """Trend regime: 1.0 when yesterday's close > its SMA, else 0.0 (causal)."""
    sma = underlying_close.rolling(window).mean()
    risk_on = (underlying_close > sma).astype(float)
    return risk_on.shift(1).rename("regime_sma")     # act on prior-day signal


def hmm_regime(returns: pd.Series, n_states: int = 2, refit_every: int = 63,
               min_obs: int = 504, seed: int = 7) -> pd.Series:
    """Walk-forward 2-state Gaussian HMM; 1.0 = benign state, 0.0 = stressed.

    Causal: at each step the model is fit on past returns only, then used to
    infer the most recent hidden state. The stressed state is the one with the
    lower mean return (label-switch safe).
    """
    from hmmlearn.hmm import GaussianHMM

    r = returns.dropna()
    n = len(r)
    out = pd.Series(np.nan, index=r.index, name="regime_hmm")
    if n <= min_obs:
        return pd.Series(1.0, index=r.index, name="regime_hmm")

    model = None
    benign_state = None
    for pos in range(min_obs, n):
        if (pos - min_obs) % refit_every == 0 or model is None:
            try:
                x = r.iloc[:pos].values.reshape(-1, 1) * 100.0
                model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                    n_iter=80, random_state=seed).fit(x)
                benign_state = int(np.argmax(model.means_.ravel()))   # highest-mean = benign
            except Exception:              # noqa: BLE001
                out.iloc[pos] = 1.0
                continue
        x_now = r.iloc[:pos].values.reshape(-1, 1) * 100.0
        state_now = int(model.predict(x_now)[-1])
        out.iloc[pos] = 1.0 if state_now == benign_state else 0.0
    return out.shift(1).ffill().fillna(1.0)          # act on prior-day inference


# --------------------------------------------------------------------------- #
# HAR-RV volatility (Corsi 2009) + Time-Series Momentum (Moskowitz+ 2012)      #
# --------------------------------------------------------------------------- #
def garman_klass_rv(ohlc: pd.DataFrame) -> pd.Series:
    """Daily realized-variance proxy from OHLC (Garman-Klass).

    HAR-RV ideally consumes intraday realized variance; with daily bars only,
    the range-based Garman-Klass estimator is a far more efficient daily variance
    proxy than squared close-to-close returns.
    """
    o, h, l, c = (np.log(ohlc[x]) for x in ("open", "high", "low", "close"))
    rv = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    return rv.clip(lower=1e-10).rename("rv")


def har_vol(ohlc: pd.DataFrame, refit_every: int = 21, min_obs: int = 252,
            trading_days: int = 252) -> pd.Series:
    """Walk-forward HAR-RV (Corsi 2009): forecast next-day vol (annualized).

    RV_t = b0 + b_d·RV_{t-1} + b_w·mean(RV_{t-5..t-1}) + b_m·mean(RV_{t-22..t-1}).
    Coefficients are re-estimated every `refit_every` days on the expanding
    window; the forecast each day uses only past RV — strictly causal.
    """
    rv = garman_klass_rv(ohlc).dropna()
    rv_d = rv.shift(1)
    rv_w = rv.rolling(5).mean().shift(1)
    rv_m = rv.rolling(22).mean().shift(1)
    X = pd.concat([rv_d, rv_w, rv_m], axis=1).dropna()
    y = rv.reindex(X.index)
    n = len(X)
    out = pd.Series(np.nan, index=X.index, name="har_vol")
    if n <= min_obs:
        return np.sqrt(rv * trading_days).rename("har_vol")

    Xv = np.column_stack([np.ones(n), X.values])     # design matrix with intercept
    yv = y.values
    beta = None
    for pos in range(min_obs, n):
        if (pos - min_obs) % refit_every == 0 or beta is None:
            beta, *_ = np.linalg.lstsq(Xv[:pos], yv[:pos], rcond=None)
        fc = float(Xv[pos] @ beta)                   # forecast RV for day `pos`
        out.iloc[pos] = np.sqrt(max(fc, 1e-10) * trading_days)
    return out.ffill()


def tsmom_signal(close: pd.Series, lookback: int = 252) -> pd.Series:
    """Time-series momentum (Moskowitz-Ooi-Pedersen 2012): 1.0 long / 0.0 flat.

    Long when the trailing `lookback`-day (≈12-month) return is positive. Causal:
    the signal acts on the next day. Long-only (flat, not short) for a long LETF.
    """
    trailing = close / close.shift(lookback) - 1.0
    return (trailing > 0).astype(float).shift(1).rename("tsmom")


def rolling_ar1(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling lag-1 autocorrelation of returns (causal, shifted).

    From Bandi-style 'beyond volatility drag' (arXiv 2504.20116): a daily-reset
    LETF beats its naive multiple when returns are positively autocorrelated
    (trending) and lags it when negatively autocorrelated (mean-reverting). This
    estimates that AR(1) coefficient over a trailing window.
    """
    def _ac(x: np.ndarray) -> float:
        a, b = x[:-1], x[1:]
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    r = returns.dropna()
    ac = r.rolling(window).apply(_ac, raw=True)
    return ac.shift(1).rename("ar1")          # act on prior-day estimate


def iv_floored_vol(har_vol: pd.Series, vix_close: pd.Series, leverage: float,
                   beta: float = 0.65, trading_days: int = 252) -> pd.Series:
    """IV-floored HAR vol: max(HAR forecast, beta * L * implied-vol). (Panel rank 2.)

    HAR/GARCH are backward-looking and under-forecast going INTO scheduled events
    (FOMC/CPI/earnings) — the 1-2 days a 3x LETF is most exposed. Flooring the
    sizer's vol with a fraction of forward implied vol (VIX, scaled to the LETF by
    its leverage) lifts the forecast on exactly those days. `beta` is tuned so the
    floor binds only ~10-20% of days; the literal max(HAR, VIX) would bind almost
    daily and degenerate into a pure (and over-conservative) VIX-target.
    """
    iv_letf = (vix_close / 100.0) * float(leverage)               # VIX points -> LETF decimal vol
    iv_letf = iv_letf.reindex(har_vol.index).ffill().shift(1)     # causal: prior-day implied vol
    floor = beta * iv_letf
    # Only FLOOR an existing HAR estimate — never substitute one during HAR's
    # warmup. Otherwise the comparison vs vol_target_har is confounded by IV
    # having no warmup (it would trade the 1999 rally HAR sits out), which inflates
    # the result for reasons unrelated to event-flooring. NaN stays NaN -> cash.
    blended = har_vol.copy()
    both = har_vol.notna() & floor.notna()
    blended[both] = np.maximum(har_vol[both], floor[both])
    return blended.rename("iv_blended_vol")
