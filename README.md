# IQIC 2026: Volatility-Targeted Inverse-Volatility Portfolio (VTRP)

Weights are inverse-volatility ("risk-balanced"). This is not exact equal-risk-contribution
risk parity, because correlations also affect each asset's risk contribution.

Code and data steps behind our IQIC 2026 Strategy Declaration. Anyone can rebuild the data,
rerun the backtest and recompute each week's target weights from public prices.

## The strategy in five rules

Every Friday close (Oct 30, Nov 6, Nov 13), traded the next Monday:

1. **Volatility:** σᵢ = standard deviation of the last 63 daily returns × √252
2. **Risk parity:** wᵢ = (1/σᵢ) ÷ Σⱼ(1/σⱼ), capped at 25% per ETF (or the StockTrak class limit if lower)
3. **Volatility target:** k = min(1, 10% ÷ σₚ), where σₚ = √(wᵀΣw) from the 63-day covariance matrix
4. **Circuit breaker:** c = 0.5 if the account is more than 8% below its peak at a Friday close, otherwise 1.
   Final targets are c × k × wᵢ, and SHY holds the rest, up to StockTrak's class position limit;
   anything above that limit stays as cash (in the backtest this lowers Sharpe from 0.644 to 0.634).
5. **Band:** trade only if some weight is 2 percentage points or more from target; then rebalance everything

Universe: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, IEF, GLD; SHY is the cash sleeve.

## Reproduce the results

```bash
pip install pandas numpy pyarrow
python build_prices.py            # downloads the data and writes prices.csv
python vtrp.py --prices prices.csv
```

With the data snapshot used in the declaration (prices through 2026-10-28), `vtrp.py` prints:

| Period | Portfolio | CAGR | Vol | Sharpe | Beta | Alpha | Treynor | Max DD |
|---|---|---|---|---|---|---|---|---|
| 2005–2015 (in-sample) | VTRP | 7.8% | 9.2% | 0.62 | 0.37 | +3.2% | 0.153 | −19.9% |
| 2005–2015 | SPY | 7.1% | 20.3% | 0.33 | 1.00 | 0 | 0.066 | −55.2% |
| 2016–2026 (out-of-sample) | VTRP | 7.5% | 8.9% | 0.67 | 0.43 | +0.1% | 0.140 | −16.6% |
| 2016–2026 | SPY | 15.0% | 17.7% | 0.78 | 1.00 | 0 | 0.138 | −33.7% |

Backtest conventions: signals at each week's last close, executed at the next trading day's
close; 5 bp cost per unit of value traded (`--cost-bp` changes it); SHY's daily return is the
risk-free rate; alpha and beta are measured against SPY. Sharpe = (Rp − Rf) ÷ σp;
β = Cov(rp, rSPY) ÷ Var(rSPY); Jensen's α = (Rp − Rf) − β(RSPY − Rf); Treynor = (Rp − Rf) ÷ β.

**Data:** Yahoo Finance split- and dividend-adjusted closes, published by the
[sp500-data](https://github.com/chan1001/sp500-data) project as the release file
`prices.parquet` (hosted at github.com/Johnbrick123/sp500-data). The declaration used the file
with SHA-256 `974f5745443a01f8bc965cc6213edf3b1f5c41fd08e02c77b6e2339ecae3531a`; the
resulting `prices.csv` has SHA-256 `bc1f943e9e847bfcb2cf2fee6d3c2acddbd4d593fb5e1e7400f3a628aa597e0f`.
The opening weights in the declaration add the Oct 29 close from the release downloaded Oct 31, 2026
(SHA-256 `65d2a2df81f23db4ccf40e52023b0fd853f486ff4aa4ef477acd207df0217d26`); its earlier prices are
identical to the snapshot above.
The release updates nightly, so a fresh download adds later days and can shift adjusted
prices slightly.

## Automated tests

```bash
python test_vtrp.py        # or: pytest test_vtrp.py   (about 10 seconds)
```

14 checks, all passing on the declared snapshot:

| Area | What is checked |
|---|---|
| No look-ahead | Weights at a date do not change when every later price is scrambled; backtest returns up to a date do not change when later prices change |
| Declared rules | On 50+ dates since 2006: long-only, weights sum to 1, no ETF above 25%, predicted volatility ≤ 10%; a tighter class cap is respected |
| Calculations | σₚ equals √(252 · wᵀΣ_daily w) computed independently; 63-day volatilities match; risk shares sum to 1 |
| Data handling | An ETF with a missing price in its 63-day window, or without enough history (XLC before 2018), gets no weight |
| Circuit breaker | Halves every risky weight below −8%, changes nothing above it |
| Position limits | With the breaker on, SHY (~50%) is capped at a 25% class limit and the rest stays as cash; risky weights unchanged |
| Orders | Never borrow (50 random accounts and caps); never buy above target; band blocks small rebalances; sells come before buys |
| Reproducibility | Sharpe, beta, alpha and max drawdown match the declaration exactly for all three periods |

The suite was itself checked by injecting three bugs (using tomorrow's price, dropping the ×252
annualisation, letting orders borrow); each one made the relevant test fail.

## Risk contributions

`python vtrp.py --prices prices.csv --weights-only` also prints each ETF's share of portfolio
risk, RCᵢ = wᵢ(Σw)ᵢ ÷ wᵀΣw. Inverse-volatility weights equalise these shares only if all
correlations are equal. At the Oct 29, 2026 close they ranged from −2.0% (XLE, which offset the
rest) to 11.3% (XLF). The weekly `signal_<date>.json` records them too.

## Replay of every historical 3-week contest under the live rules

```bash
python contest_sim.py --prices prices.csv     # about 3 minutes; writes contests.csv
```

Each contest starts with $100,000 cash after a Friday close, builds on Monday, can rebalance on
the next two Mondays (2-point band), and ends on the 15th trading day. Whole shares, 10 bp per
trade, uninvested cash earns nothing, and the circuit breaker uses the account's peak since the
contest began, exactly as declared for StockTrak. Trades are modelled at Monday's close.

| 1,105 contests | 2005–2015 | 2016–2026 |
|---|---|---|
| Alpha positive | 50.5% | 50.2% |
| Sharpe above SPY's | 46% | 42% |
| Loss when SPY fell, % of SPY's | 44% | 51% |
| Breaker fired | 0.2% | 0.7% |

With 10 bp costs, 50.5% and 50.2% are within one standard error (about 2 points) of 50%, so the
alpha condition is unconfirmed under live rules. Costs drive that result: across all 1,105
contests, alpha is positive in 52.9% at 5 bp, 54.3% at 2 bp and 55.8% at 0 bp per trade.

The script also reports sensitivity runs (no breaker; 20, 5, 2 and 0 bp costs; volatility
targets 8/12/14%).

## Weekly trading on StockTrak

```bash
python vtrp.py --prices prices.csv --weights-only        # target weights at the last close
python order_worksheet.py --prices prices.csv --account account.json \
    --quotes quotes.csv --signal-date 2026-10-30
```

- `account.json`: copy equity, cash and share holdings from StockTrak (My Portfolio → Portfolio
  Summary and Open Positions). Set `max_position_weight` to the class position limit,
  `commission_per_trade` to the class fee, `nav_history` to the account's closing values since
  Nov 2, and `first_build` to `true` only for Nov 2. See `examples/account_example.json`.
- `quotes.csv`: Monday's bid and ask for each ETF, after 10:00 ET. See `examples/quotes_example.csv`
  (those prices are Oct 28 closes ± $0.01, for illustration only).
- Output: `orders_<date>.csv` lists sells first, then buys, then SHY. Enter each as a limit order at
  `limit` (the bid–ask midpoint); at 15:30 ET re-price anything unfilled to `reprice_1530`.
  `signal_<date>.json` is the audit record: weights, volatilities, k, drawdown and breaker state.
- The worksheet never places orders. Update `account.json` from StockTrak's actual fills.
- Audit trail: record every order in `examples/trade_log_template.csv` (time, limit, fill, fee,
  target weight, triggering rule, exceptions) and each week in `examples/weekly_review_template.csv`
  (equity, peak, drawdown, breaker state, largest weight gap, fills, actual vs target).

## Research record

`research/` holds the exploratory scripts behind every test reported in the declaration,
kept unedited so the record is complete, including the designs we rejected:

| Script | What it tested |
|---|---|
| `engine.py` | Original backtest engine (the final strategy matches `vtrp.py` to 1e-16 per day) |
| `grid.py` | Sector momentum, trend filters, top-N, momentum tilts, bands, breakers (2002–2015) |
| `multi.py` | Sectors only vs adding IEF, GLD, TLT (chose sectors + IEF + GLD on 2005–2015) |
| `extras.py` | Robustness grid (12 settings), 20 bp costs, no volatility target, stress periods |
| `analyze.py` | Rolling 15-trading-day windows |
| `their.py` | 20- and 5-day relative-momentum rule (rejected) |
| `compare.py` | Top-3 sector rotation with SPY core vs our strategy (rejected) |

These scripts expect a long-format `data/etf.csv` (date, ticker, adj_close) in the parent folder,
the format they were written against. `compare.py` also needs the rotation candidate's own
script (`IQIC_2026_Sector_Alpha_Backtest.py`) and its outputs, which were run unmodified on the
same prices.
