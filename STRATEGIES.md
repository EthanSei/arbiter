# Strategies

## Next Steps (Ordered by Priority)

1. **Fix Fee Model** — Replace flat `fee_rate=0.01` with Kalshi's actual parabolic formula `0.07 * P * (1-P)`. Current flat rate underestimates fees by ~50% in the 50-70c range. All backtests are unreliable until this is fixed.
2. **Expand Structural Arbitrage** — Extend ConsistencyStrategy beyond MAXMON/MINMON to all bracket families, complementary contracts that should sum to 1.0, and conditional probability constraints across categories.
3. **Informed Market-Making** — Post limit orders (as maker) where model identifies probability edge. Captures maker fee advantage (4x lower than taker) + model edge. Requires authenticated trading API, order management, websocket feeds.
4. **Collect Historical Data at Scale** — Scrape all Kalshi settled markets + candlesticks (not just econ indicators). Train actual LightGBM model and run honest backtests with real fees.
5. **Favorite-Longshot Bias** — Systematically buy favorites (>50c), avoid longshots (<10c). Documented on 300K+ Kalshi contracts (Whelan & Bürgi 2025), but edge is thin (1.9% post-fee, makers only) and shrinking as institutional market makers (SIG) enter.

---

## Implemented Strategies

### EVStrategy / YesOnlyEVStrategy

**Summary:** Per-contract expected value using LightGBM probability estimates vs market price.

**Files:** `arbiter/scoring/strategy.py:49-88`, `arbiter/scoring/ev.py`

**How it works:** For each contract, calls `estimator.estimate()` to get model probability, then:
```
EV = model_prob - (market_price + fee_rate)
```
Scores both YES and NO sides. `YesOnlyEVStrategy` filters to YES-only because backtests showed the model is anti-calibrated on NO-side trades (46.8% win rate vs 59.7% breakeven needed).

**Backtest results (YES-side):** 48.2% win rate vs 41.1% breakeven — apparent 7.1pp edge.

**Drawbacks:**
- Model weights (`arbiter_lgbm.pkl`) don't exist — estimator falls back to market midpoint, producing EV = -fee_rate always
- Backtest results are unreliable: flat `fee_rate=0.01` underestimates real Kalshi fees by ~50%
- Anti-calibrated on NO-side trades (hence YES-only variant)
- Requires a trained model with good calibration to produce any signal

---

### ConsistencyStrategy

**Summary:** Detects stochastic dominance violations in Kalshi range-market bracket families (MAXMON/MINMON). Pure structural arb — no model needed.

**Files:** `arbiter/scoring/strategy.py:91-109`, `arbiter/scoring/consistency.py`

**How it works:** For MAXMON (above) / MINMON (below) bracket families, enforces that P(above X1) >= P(above X2) when X1 <= X2. Uses sibling contract prices as probability floors:
```
EV = sibling_price - contract.yes_price - fee_rate
```
Ignores the ML estimator entirely. Only produces YES-direction opportunities.

**Drawbacks:**
- Low frequency — violations are rare and may be fleeting
- Only covers MAXMON/MINMON tickers currently (not all bracket families)
- Other bots likely detect the same violations
- Limited to Kalshi range markets

**Strengths:**
- Mathematically certain edge (no model risk)
- Fee-insensitive (arb profit exceeds fees when it fires)
- Already implemented and tested

---

### AnchorStrategy

**Summary:** Compares Kalshi economic indicator contract prices against external probability anchors derived from FRED/BLS forecast data using a normal distribution model.

**Files:** `arbiter/scoring/strategy.py:112-176`, `arbiter/scoring/anchor.py`

**How it works:** For each economic indicator group (e.g., KXCPI), loads consensus forecast (μ) and historical surprise volatility (σ) from FRED/BLS providers, then:
```
anchor_prob = P(X > K) = 1 - Φ((K - μ) / σ)
```
Optionally applies PlattCalibrator (per-series logistic regression on log-odds). Flags contracts where `anchor_prob > market_price + fee_rate`.

**Backtest results:**
| Config | Brier Score | Accuracy |
|---|---|---|
| Raw baseline | 0.1261 | 88.5% |
| MAD-winsorized + Platt calibration | 0.0524 | 92.7% |

Win rate on high-edge signals: 80.5% (breakeven = 63.2%) across 1,318 settled records.

**Drawbacks:**
- **Naive consensus** — uses last month's value or 4-week moving average, NOT real analyst survey medians (Bloomberg/Reuters)
- **No time-to-release awareness** — treats stale μ as equally valid whether release is 3 weeks or 3 hours away
- **Small sample sizes** — monthly indicators give ~12 obs/year for σ estimation
- **Normal distribution assumption** — fat tails mean underestimating extreme threshold probabilities
- **Limited coverage** — only 5 indicator series (CPI, CPI YoY, Core CPI YoY, Payrolls, Jobless Claims)

See [anchor_limitations.md](anchor_limitations.md) in memory for full analysis.

---

### IndicatorRouter

**Summary:** Routes contracts by ticker prefix — indicator contracts go to ConsistencyStrategy + AnchorStrategy; everything else goes to YesOnlyEVStrategy.

**File:** `arbiter/scoring/strategy.py:234-273`

**Rationale:** LightGBM EV is excluded for indicator contracts because the model has no current-cycle economic release knowledge. This is the default production router.

---

## Strategies Investigated but Not Implemented

### Price Momentum / Mean-Reversion

**Summary:** Predict short-term price movements (T+2h, T+24h) on binary contracts using technical features from hourly candlestick data.

**Verdict: Not viable.**

**Evidence against:**
- Academic analysis (Altos & Gorbatov 2024) shows **negative autocorrelation** in prediction market price changes — momentum would systematically lose
- Prediction markets are fundamentally different from equities: prices bounded [0,1], converge to 0/1 at expiry, driven by information arrival not supply/demand
- Professional market makers now dominate Kalshi orderbooks; spreads compressed from 4.5% (2023) to 1.2% (2025)
- Traditional TA indicators (RSI, MACD, Bollinger) assume unbounded, mean-reverting prices — opposite of prediction market dynamics

---

### Favorite-Longshot Bias (FLB)

**Summary:** Systematically buy contracts priced >50c (favorites) and avoid/sell contracts <10c (longshots), exploiting the documented tendency for longshots to be overpriced.

**Verdict: Real but thin edge, not recommended as primary strategy.**

**Evidence for:**
- Whelan & Bürgi (2025): 300K+ Kalshi contracts confirm the bias. Contracts <10c lose 60%+ of capital; contracts >50c earn small positive returns
- NBER WP 15923: Theoretical foundation — noise traders with imprecise signals systematically overpay for longshots
- Bias is strongest in thin, retail-heavy markets (entertainment, weather, niche)

**Evidence against:**
- Returns are only **2.6% pre-fee, 1.9% post-fee — and only for makers** (limit orders). Takers lose money across all price buckets
- Kalshi's parabolic fee formula peaks at 50c — exactly where this strategy concentrates trades
- Bias is documented as **shrinking over time** as institutional market makers (SIG) enter
- Requires maker execution (limit orders + fill management), not the scan-and-alert taker model our codebase implements
- At realistic capital deployment, generates perhaps $50-200/week — marginal as side income

---

### Cross-Platform Arbitrage

**Summary:** Exploit price discrepancies on identical events between Kalshi and Polymarket.

**Verdict: Impractical for us.**

**Evidence against:**
- Both platforms return empty `category=""` and cover different market niches — matching is unreliable
- Spreads compressed to ~1.2% as of 2025; arb opportunities last seconds to minutes
- Requires simultaneous positions on both platforms (capital-intensive)
- Polymarket uses crypto settlement (slower, gas fees)

---

### Event-Driven / News Sentiment

**Summary:** Use news/sentiment signals to predict price movements ahead of the market.

**Status: Not investigated in depth.**

**Potential:** The $85M Polymarket trader (Théo) profited from commissioning original polling research — pure information arbitrage. NLP/sentiment could provide a weaker version of this.

**Concerns:** Requires real-time news feeds, NLP infrastructure, and speed to beat other bots. High complexity for uncertain edge.

---

## Key Infrastructure Issues

### Fee Model is Wrong
The codebase uses flat `fee_rate=0.01` (1%). Kalshi's actual formula:
```
taker_fee = roundup(0.07 * contracts * P * (1-P))   # max 1.75c at P=0.50
maker_fee = roundup(0.0175 * contracts * P * (1-P))  # 4x cheaper
```
At 60c (where favorites concentrate), actual taker fee is ~2.8% — nearly 3x the modeled rate. **All backtests overstate edge until this is fixed.**

### No Trained Model
`models/arbiter_lgbm.pkl` doesn't exist. LGBMEstimator falls back to market midpoint. EVStrategy produces no useful signal. Need to collect settlement data across all Kalshi series and train an actual model.

### Maker vs Taker Architecture Gap
The codebase is a scan-and-alert system (identify mispricing → send Discord alert → human takes action). This is inherently taker execution. The documented edges (FLB, market-making) require maker execution (posting limit orders). Bridging this gap requires: authenticated Kalshi trading API, order management, websocket feeds for real-time updates, and inventory/position tracking.
