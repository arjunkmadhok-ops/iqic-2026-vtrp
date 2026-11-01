#!/usr/bin/env python3
"""Automated checks for vtrp.py and order_worksheet.py.

  python test_vtrp.py            # or: pytest test_vtrp.py

Needs prices.csv in this folder (build_prices.py). The reproduction test runs only when
prices.csv is the exact snapshot used in the Strategy Declaration (checked by SHA-256).
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import order_worksheet as ow
import vtrp

HERE = Path(__file__).parent
PX = vtrp.load_prices(HERE / "prices.csv")
DECLARED_SHA = "bc1f943e9e847bfcb2cf2fee6d3c2acddbd4d593fb5e1e7400f3a628aa597e0f"
TOL = 1e-12


def _week_ends(px, start="2006-01-01", step=25):
    idx = px.index
    we = sorted(set(pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max()))
    return [d for d in we if d >= pd.Timestamp(start)][::step]


# --- No look-ahead -------------------------------------------------------------------------
def test_weights_ignore_future_prices():
    """Targets at date t must not change when every later price is scrambled."""
    rng = np.random.default_rng(0)
    for d in _week_ends(PX, step=150):
        loc = PX.index.get_loc(d)
        scrambled = PX.copy()
        scrambled.iloc[loc + 1:] *= rng.uniform(0.5, 1.5, size=scrambled.iloc[loc + 1:].shape)
        a, _ = vtrp.risk_parity_targets(PX, loc)
        b, _ = vtrp.risk_parity_targets(scrambled, loc)
        assert (a - b).abs().max() < TOL, d


def test_backtest_ignores_future_prices():
    """Daily backtest returns up to date T must not change when prices after T change."""
    cut = PX.index.get_loc(pd.Timestamp("2025-06-30"))
    r_full, _, _ = vtrp.backtest(PX, start="2024-01-02")
    r_cut, _, _ = vtrp.backtest(PX.iloc[:cut + 1], start="2024-01-02")
    common = r_cut.index
    assert (r_full.loc[common] - r_cut).abs().max() < TOL


# --- Weights obey every declared rule -------------------------------------------------------
def test_weights_valid_on_many_dates():
    for d in _week_ends(PX):
        loc = PX.index.get_loc(d)
        w, info = vtrp.risk_parity_targets(PX, loc)
        assert (w >= -TOL).all(), d                                  # long only
        assert abs(w.sum() - 1) < 1e-9, d                            # fully allocated, no leverage
        assert (w[vtrp.RISKY] <= vtrp.PARAMS["cap"] + 1e-9).all(), d  # 25% cap
        names = list(info["cov"].index)
        realised = math.sqrt(w[names].values @ info["cov"].values @ w[names].values)
        assert realised <= vtrp.PARAMS["target_vol"] + 1e-9, d       # 10% volatility target


def test_annualisation_matches_daily_formula():
    """sigma_p must equal sqrt(252 * w' Sigma_daily w) computed independently."""
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    w, info = vtrp.risk_parity_targets(PX, loc)
    names = list(info["cov"].index)
    daily = PX[names].iloc[loc - 63: loc + 1].pct_change().iloc[1:]
    ws = w[names] / info["k"]                                       # weights before scaling by k
    independent = math.sqrt(252 * ws.values @ daily.cov().values @ ws.values)
    assert abs(independent - info["pred_vol"]) < 1e-12
    vol = daily.std() * math.sqrt(252)
    assert (vol - info["vol"][names]).abs().max() < 1e-12


def test_tighter_cap_is_respected():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    w, _ = vtrp.risk_parity_targets(PX, loc, cap=0.10)
    assert (w[vtrp.RISKY] <= 0.10 + 1e-9).all()
    assert abs(w.sum() - 1) < 1e-9


def test_missing_data_makes_etf_ineligible():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    gap = PX.copy()
    gap.iloc[loc - 10, gap.columns.get_loc("XLK")] = np.nan
    w, _ = vtrp.risk_parity_targets(gap, loc)
    assert w["XLK"] == 0 and abs(w.sum() - 1) < 1e-9


def test_etf_not_used_before_it_has_history():
    loc = PX.index.get_loc(pd.Timestamp("2017-06-30"))        # XLC launched June 2018
    w, _ = vtrp.risk_parity_targets(PX, loc)
    assert w["XLC"] == 0 and w["XLRE"] > 0


def test_circuit_breaker():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    w, _ = vtrp.risk_parity_targets(PX, loc)
    off = vtrp.apply_breaker(w, -0.07)
    on = vtrp.apply_breaker(w, -0.09)
    assert (off - w).abs().max() < TOL
    assert (on[vtrp.RISKY] - 0.5 * w[vtrp.RISKY]).abs().max() < TOL
    assert abs(on.sum() - 1) < 1e-9


def test_shy_respects_class_position_limit():
    """With the breaker on, SHY would be ~50%; a 25% class limit must cap it, rest stays cash."""
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    w, _ = vtrp.risk_parity_targets(PX, loc)
    w = vtrp.apply_breaker(w, -0.10)
    capped, cash_w = vtrp.cap_cash_sleeve(w, 0.25)
    assert capped[vtrp.CASH] <= 0.25 + 1e-12 and cash_w > 0.2
    assert abs(capped.sum() + cash_w - 1) < 1e-9
    assert (capped[vtrp.RISKY] - w[vtrp.RISKY]).abs().max() < TOL      # risky weights untouched
    q = _quotes(loc)
    equity = 100_000.0
    orders, cash, _, _ = ow.plan_orders(capped, {}, equity, equity, q, first=True)
    shy = sum(o["shares"] * q[vtrp.CASH][1] for o in orders if o["ticker"] == vtrp.CASH)
    assert shy <= 0.25 * equity + 1e-6 and cash >= cash_w * equity - 1_000


def test_risk_contributions_sum_to_one():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    w, info = vtrp.risk_parity_targets(PX, loc)
    rc = vtrp.risk_contributions(w, info["cov"])
    assert abs(rc.sum() - 1) < 1e-9


# --- Order arithmetic ------------------------------------------------------------------------
def _quotes(loc):
    last = PX.iloc[loc]
    return {a: (round(last[a] - 0.01, 2), round(last[a] + 0.01, 2)) for a in vtrp.RISKY + [vtrp.CASH]}


def test_orders_never_borrow_and_respect_targets():
    rng = np.random.default_rng(1)
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    q = _quotes(loc)
    for _ in range(50):
        w, _ = vtrp.risk_parity_targets(PX, loc, cap=float(rng.uniform(0.08, 0.25)))
        equity = float(rng.uniform(20_000, 500_000))
        orders, cash, _, trade = ow.plan_orders(w, {}, equity, equity, q, fee=float(rng.choice([0, 5])),
                                                first=True)
        assert trade and cash >= -1e-6
        for o in orders:
            assert o["action"] == "BUY" and o["shares"] > 0
            assert o["shares"] * ow.mid(q[o["ticker"]]) <= equity * w[o["ticker"]] + 1e-6


def test_band_blocks_small_rebalances():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    q = _quotes(loc)
    w, _ = vtrp.risk_parity_targets(PX, loc)
    equity = 100_000.0
    orders, cash, _, _ = ow.plan_orders(w, {}, equity, equity, q, first=True)
    held = {o["ticker"]: o["shares"] for o in orders}
    again, _, gap, trade = ow.plan_orders(w, held, cash, equity, q)
    assert not trade and not again and gap < vtrp.PARAMS["band"]


def test_sells_come_before_buys():
    loc = PX.index.get_loc(pd.Timestamp("2026-10-23"))
    q = _quotes(loc)
    w, _ = vtrp.risk_parity_targets(PX, loc)
    held = {"XLK": 200, "SHY": 300}                  # overweight XLK and SHY, nothing else
    equity = 200 * ow.mid(q["XLK"]) + 300 * ow.mid(q["SHY"]) + 50_000
    orders, cash, _, trade = ow.plan_orders(w, held, 50_000, equity, q)
    actions = [o["action"] for o in orders]
    assert trade and actions == sorted(actions, key=lambda a: a != "SELL") and cash >= -1e-6


# --- Reproducibility --------------------------------------------------------------------------
def test_reproduces_declared_results():
    sha = hashlib.sha256((HERE / "prices.csv").read_bytes()).hexdigest()
    if sha != DECLARED_SHA:
        print("  (skipped: prices.csv is not the declared snapshot)")
        return
    r, _, T = vtrp.backtest(PX)
    rets = PX.pct_change(fill_method=None)
    d = pd.DataFrame({"s": r, "b": rets["SPY"], "rf": rets["SHY"]}).loc["2005-08-01":].fillna(0)
    expected = {("2005-08-01", "2015-12-31"): (0.619, 0.370, 0.032, -0.199),
                ("2016-01-01", "2026-12-31"): (0.669, 0.425, 0.001, -0.166),
                ("2005-08-01", "2026-12-31"): (0.644, 0.394, 0.018, -0.199)}
    for (a, b), (sh, beta, alpha, mdd) in expected.items():
        x = d.loc[a:b]
        s = vtrp.stats(x["s"], x["b"], x["rf"])
        got = (round(s["Sharpe"], 3), round(s["Beta"], 3), round(s["Alpha"], 3), round(s["MaxDD"], 3))
        assert got == (sh, beta, alpha, mdd), (a, b, got)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
