"""Backtest engine for the IQIC Strategy Declaration.

Strategy: Volatility-Targeted Sector Momentum (VTSM)
  1. Universe: SPDR Select Sector ETFs (XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY),
     each eligible once it has LOOKBACK+SKIP trading days of history.
  2. Signal (computed at the last close of each week):
       score_i = total return from t-LOOKBACK to t-SKIP  /  annualised 63-day volatility
  3. Selection: top N scores, and only sectors whose price is above their 100-day SMA.
  4. Weights inside the equity sleeve: inverse 63-day volatility, capped at CAP per ETF.
  5. Exposure: scale the sleeve so predicted portfolio volatility (63-day covariance)
     = TARGET_VOL, max 100% equity. The remainder is held in SHY (1-3y Treasuries).
  6. Trades are executed one full trading day after the signal (close t+1); cost COST per unit turnover.
"""
import numpy as np
import pandas as pd

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
CASH = "SHY"
BENCH = "SPY"


def load(path):
    df = pd.read_csv(path)
    px = df.pivot(index="d", columns="ticker", values="adj_close")
    px.index = pd.to_datetime(px.index)
    return px.sort_index()


def week_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby(idx.to_period("W-FRI")).max().values)


def target_weights(px, rets, t, P):
    """Weights decided with data up to and including date t."""
    loc = px.index.get_loc(t)
    lb, skip, n = P["lookback"], P["skip"], P["top_n"]
    if loc < lb + skip + 5:
        return None
    window = rets.iloc[loc - P.get("vol_win", 63) + 1: loc + 1]
    vol = window.std() * np.sqrt(252)
    scores = {}
    for s in SECTORS:
        p_now = px[s].iloc[loc - skip]
        p_then = px[s].iloc[loc - lb]
        if np.isnan(p_now) or np.isnan(p_then) or window[s].isna().any():
            continue
        mom = p_now / p_then - 1
        scores[s] = mom / vol[s] if P["risk_adj"] else mom
    if not scores:
        return None
    if P.get("tilt") is not None:
        sc = pd.Series(scores)
        z = (sc - sc.mean()) / sc.std()
        names = list(sc.index)
        iv = (1 / vol[names]) * (1 + P["tilt"] * z.clip(-1.5, 1.5)).clip(lower=0)
        ws = iv / iv.sum()
        for _ in range(20):
            over = ws > P["cap"] + 1e-12
            if not over.any():
                break
            excess = (ws[over] - P["cap"]).sum(); ws[over] = P["cap"]
            under = ws < P["cap"] - 1e-12
            ws[under] += excess * ws[under] / ws[under].sum()
        cov = window[names].cov() * 252
        pvol = float(np.sqrt(ws.values @ cov.values @ ws.values))
        scale = min(P["max_eq"], P["target_vol"] / pvol) if P["target_vol"] else P["max_eq"]
        w = pd.Series(0.0, index=SECTORS + [CASH]); w[names] = ws * scale
        w[CASH] = 1 - w.sum()
        return w
    ranked = sorted(scores, key=scores.get, reverse=True)[:n]
    if P["trend"]:
        sma = px[ranked].iloc[loc - P["sma"] + 1: loc + 1].mean()
        ranked = [s for s in ranked if px[s].iloc[loc] > sma[s]]
    w = pd.Series(0.0, index=SECTORS + [CASH])
    if ranked:
        iv = 1 / vol[ranked]
        ws = iv / iv.sum()
        # cap and redistribute
        for _ in range(10):
            over = ws > P["cap"]
            if not over.any():
                break
            excess = (ws[over] - P["cap"]).sum()
            ws[over] = P["cap"]
            under = ~over
            ws[under] += excess * ws[under] / ws[under].sum()
        # if fewer than 1/cap names, unfilled weight stays unallocated (goes to cash)
        ws = ws.clip(upper=P["cap"])
        cov = window[ranked].cov() * 252
        pvol = float(np.sqrt(ws.values @ cov.values @ ws.values))
        scale = min(P["max_eq"], P["target_vol"] / pvol) if P["target_vol"] else P["max_eq"]
        scale = min(scale, 1.0 / ws.sum())  # never above 100% equity
        w[ranked] = ws * scale
    w[CASH] = 1 - w.sum()
    return w


DEFAULT = dict(lookback=126, skip=5, top_n=4, risk_adj=True, trend=True, sma=100,
               cap=0.30, target_vol=0.10, max_eq=1.0, cost=0.0005, dd_breaker=None, tilt=None, band=None)


def run(px, P=DEFAULT, start=None, end=None):
    P = {**DEFAULT, **P}
    rets = px.pct_change(fill_method=None)
    idx = px.index
    if start:
        idx = idx[idx >= pd.Timestamp(start)]
    if end:
        idx = idx[idx <= pd.Timestamp(end)]
    wends = week_ends(px.index)
    cols = SECTORS + [CASH]
    cur = pd.Series(0.0, index=cols); cur[CASH] = 1.0
    pending = None
    out_r, out_w, turn = [], [], []
    for i, t in enumerate(idx):
        r = rets.loc[t, cols].fillna(0.0)
        # return earned today on yesterday's holdings
        day_r = float((cur * r).sum())
        # drift
        grown = cur * (1 + r)
        cur = grown / grown.sum() if grown.sum() != 0 else cur
        tc = 0.0
        if pending is not None:            # execute at today's close (decided yesterday)
            tw = pending.reindex(cols).fillna(0.0)
            if P.get("band") and float((tw - cur).abs().max()) < P["band"]:
                tw = cur
            to = float((tw - cur).abs().sum())
            tc = to * P["cost"]
            turn.append((t, to))
            cur = tw
            pending = None
        out_r.append(day_r - tc)
        out_w.append(cur.copy())
        if t in wends:
            w = target_weights(px, rets, t, P)
            if w is not None:
                if P.get("dd_breaker"):
                    eq = np.cumprod(1 + np.array(out_r[-63:]))
                    dd = eq[-1] / max(1.0, eq.max()) - 1
                    if dd < -P["dd_breaker"]:
                        w = w.copy(); w[SECTORS] *= 0.5; w[CASH] = 1 - w[SECTORS].sum()
                pending = w
    res = pd.Series(out_r, index=idx, name="strategy")
    W = pd.DataFrame(out_w, index=idx)
    T = pd.Series(dict(turn))
    return res, W, T


def stats(r, b, rf):
    """Annualised stats of daily returns r vs benchmark b, risk-free rf (daily)."""
    ex = r - rf
    exb = b - rf
    ann = 252
    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = ex.mean() / r.std() * np.sqrt(ann)
    beta = np.cov(r, b)[0, 1] / b.var()
    alpha = (ex.mean() - beta * exb.mean()) * ann
    treynor = ex.mean() * ann / beta
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Beta=beta, Alpha=alpha, Treynor=treynor, MaxDD=mdd)
