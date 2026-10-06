# Pathos

**Sentiment-aware portfolio construction in Rust.** Pathos scores recent news
headlines with **FinBERT running natively in Rust** (via
[candle](https://github.com/huggingface/candle), no Python), blends that with
fundamentals pulled from **SEC EDGAR**, and feeds the result into a
**Black-Litterman** model. A long-only, position-capped mean-variance optimizer
turns that into a portfolio and whole-share orders. You can run it from the
command line or through a web dashboard.

It also tests whether the signal works. `pathos evaluate` measures predictive
power on point-in-time history and fits the views from the data. In a two-year
study of 24 large caps, headline sentiment had no detectable predictive power,
and the calibrated model correctly stayed close to the market portfolio instead
of trading on noise ([results](#results)).

![Pathos dashboard](docs/dashboard-light.png)

## How it works

```mermaid
flowchart LR
    subgraph Data
        Y["Yahoo Finance<br/>daily prices"]
        S["SEC EDGAR<br/>XBRL company facts"]
        N["Yahoo + Google News<br/>RSS headlines"]
    end
    N --> F["FinBERT in candle<br/>p+, p−, p0 per headline"]
    F --> A["Recency-weighted score<br/>+ confidence"]
    S --> Q["Quality score<br/>growth · margin · leverage"]
    A --> V["Signal → views Q, Ω"]
    Q --> V
    Y --> C["Ledoit-Wolf<br/>covariance Σ"]
    S --> M["Market caps<br/>shares × price"]
    C --> P["Equilibrium prior<br/>π = δΣw_mkt"]
    M --> P
    P --> BL["Black-Litterman<br/>posterior μ, Σ_BL"]
    V --> BL
    BL --> O["Long-only capped<br/>mean-variance (FISTA)"]
    O --> R["Report · CLI table · dashboard"]
```

1. **Data.** Prices come from the Yahoo chart API and are aligned on common
   trading days before computing log returns. Fundamentals come from SEC
   company facts and are robust to messy filings:
   * try several us-gaap tags per concept;
   * keep only true ~365-day 10-K periods;
   * de-duplicate restatements;
   * reject stale data.

   Headlines come from Yahoo Finance and Google News RSS, filtered to ones that
   actually mention the company. Every response is cached on disk.
2. **Sentiment.** `ProsusAI/finbert` (BERT-base + pooler + classifier) runs in
   candle. Weights are pinned to a specific Hub revision and SHA-256 verified.
   They were checked to be bit-identical to the official PyTorch checkpoint,
   with outputs matching `transformers` to 4 decimal places. Each headline gets
   a polarity score $s = p_{+} - p_{-}$. These are combined per ticker with a
   3-day half-life. Confidence grows with the effective number of headlines and
   shrinks when they disagree. Scores are memoized, so repeat runs skip
   inference. By default the model is quantized to int8 (see
   [Model size](#model-size)).
3. **Signal.** A confidence-weighted blend of news sentiment and a fundamentals
   quality score (revenue growth, net margin, debt/equity).
4. **Black-Litterman.** One absolute view per asset with a meaningful signal:

   $$Q_i = \pi_i + \kappa\, \text{signal}_i\, \sigma_i, \qquad \Omega_{ii} = \tau\sigma_i^2\,\frac{1-c_i}{c_i}$$

   so a signal of ±1 tilts expected return by $\kappa$ volatilities, and a
   confidence of $c = 0.5$ lands halfway between prior and view (Idzorek-style).
   The posterior is

   $$\mu_{BL} = \pi + \tau\Sigma P^\top\left(P\tau\Sigma P^\top + \Omega\right)^{-1}(Q - P\pi)$$

   solved with a Cholesky factorization rather than explicit inverses.
5. **Optimization.** The optimizer solves

   $$\max_w\ \mu^\top w - \tfrac{\delta}{2} w^\top \Sigma_{BL} w \quad \text{s.t.}\quad \textstyle\sum w_i = 1,\ 0 \le w_i \le w_{\max}$$

   using FISTA, with an exact projection onto the capped simplex. Weights are
   then rounded to whole shares with a greedy pass that spends leftover cash
   where it best reduces tracking error.

## Quick start

Requires a recent stable Rust toolchain (edition 2024).

```bash
# SEC requires a contact email in the User-Agent for EDGAR API access.
export SEC_USER_AGENT="Your Name you@example.com"

# Web dashboard on http://127.0.0.1:3000
cargo run --release -- serve

# Or the CLI
cargo run --release -- analyze AAPL MSFT NVDA GOOGL AMZN TSLA --budget 10000 --json report.json

# Score arbitrary text with FinBERT
cargo run --release -- score "Shares surge after record quarterly earnings"
```

On the first run, the official 438 MB FinBERT checkpoint is downloaded and
checksum-verified, quantized to a 117 MB int8 file in
`~/.cache/pathos/models/finbert`, and the original is deleted. A full analysis
of six tickers takes roughly 15 seconds on a laptop CPU, almost all of it
inference. Repeat runs are much faster because both HTTP responses and headline
scores are cached.

### Model size

FinBERT is BERT-base: 110 million parameters, 438 MB as 32-bit floats. Pathos
stores every weight matrix as GGML `Q8_0` (8-bit integers with one scale per
block of 32 values) and keeps biases and LayerNorm parameters in f32. candle
has no quantized BERT, so the encoder is re-implemented on its quantized
matrix multiply (`src/sentiment/qbert.rs`). `pathos benchmark-model` compares
the precisions on cached headlines; on 1,500 of them:

| `--precision` | File | Headlines/s | Same label as f32 | Score correlation | Mean \|Δ score\| |
|---|---|---|---|---|---|
| `f32` | 438 MB | 19.1 | — | — | — |
| `q8` (default) | 117 MB | 18.8 | 99.73% | 0.99992 | 0.003 |
| `q8-native` | 117 MB | 7.7 | 99.40% | 0.99979 | 0.005 |

The default `q8` keeps int8 weights on disk and expands them to f32 at load
time. candle's int8 kernels are tuned for matrix-vector LLM decoding and run
slower than its f32 GEMM for batched encoder inference, so expanding is
the faster choice. `q8-native` computes in int8 and uses about 4× less memory
at runtime. x86-64 builds enable AVX2 in `.cargo/config.toml`, without which
the int8 kernels fall back to scalar code that is roughly 10× slower.

Without `SEC_USER_AGENT` everything still works, with two fallbacks: no
fundamentals, and an equal-weight prior instead of market caps. The report
warns you when this happens.

### Docker

```bash
docker build -t pathos .
docker run -p 3000:3000 -e SEC_USER_AGENT="Your Name you@example.com" -v pathos-data:/data pathos
```

### Example output

```
Ticker      Price  Sentiment   News   Signal    Prior    Post.   Mkt Wt   Weight      Value Shares
-----------------------------------------------------------------------------------------------------
NVDA       241.60      +0.28     40    +0.39   +13.6%   +15.9%    25.2%    28.6%    2858.83     11
MSFT       531.94      +0.19     40    +0.31    +9.2%   +10.8%    17.1%    19.2%    1915.62      4
GOOGL      347.91      +0.01     40    +0.16    +9.9%   +10.9%    18.4%    17.8%    1780.57      5
AAPL       333.01      -0.07     40    +0.03    +5.5%    +5.7%    21.0%    16.3%    1628.89      5
AMZN       255.24      +0.08     40    +0.16   +10.7%   +11.9%    11.9%    11.7%    1170.67      4
TSLA       380.67      +0.13     40    +0.12   +13.8%   +15.2%     6.5%     6.5%     645.41      2

Expected excess return +11.9%  ·  Volatility 21.0%  ·  Sharpe 0.57
Invested 9972.32  ·  Cash left 27.68
```

## Configuration

| Flag / field | Default | Meaning |
|---|---|---|
| `--budget` | 10000 | Cash to allocate |
| `--max-weight` | 0.35 | Position cap (raised to 1/N if infeasible) |
| `--lookback-days` | 365 | Price history for the risk model |
| `--news-days` | 7 | Headline window |
| `--risk-aversion` | 2.5 | δ, used for the prior and the optimizer |
| `--tau` | 0.05 | Black-Litterman τ |
| `--view-scale` | 0.25 | κ, the view tilt in volatilities |
| `--sentiment-weight` | 0.7 | News vs. fundamentals share of the signal |
| `--precision` | `q8` | FinBERT weights: `q8`, `q8-native` or `f32` (any command) |
| `--keep-f32` | off | Keep the 438 MB checkpoint after quantizing |

| Environment variable | Purpose |
|---|---|
| `SEC_USER_AGENT` | `"Name email"` contact string required by SEC EDGAR |
| `PATHOS_CACHE_DIR` | HTTP cache and model location (default `~/.cache/pathos`) |
| `PATHOS_MODEL_DIR` | Override the FinBERT weights directory |
| `PATHOS_PRECISION` | Default for `--precision` |
| `RUST_LOG` | Log filter, e.g. `pathos=debug` |

The dashboard talks to a small JSON API:

* `POST /api/analyze` takes the same fields as the table above (plus
  `tickers`) and returns the full report.
* `GET /api/defaults` returns the default parameters.
* `GET /api/health` is a liveness check.

## Evaluating the signal

A sentiment signal is only worth using if it predicts returns. `pathos evaluate`
tests that on point-in-time historical data and fits the signal-to-view mapping
from the results instead of using hand-picked constants.

```bash
cargo run --release -- evaluate                  # 24 large caps, last 2 years
cargo run --release -- evaluate AAPL MSFT NVDA JPM XOM KO --years 1
cargo run --release -- evaluate --headlines-csv news.csv   # your own dated headlines
```

What it does:

1. **Collects historical headlines** from date-bounded Google News queries, one
   per ticker-week, or from a CSV with date, ticker and headline columns (for
   example [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)). Scores
   are cached, so FinBERT only runs once per headline.
2. **Avoids look-ahead.** A signal on day $t$ uses only headlines dated before
   $t$ and SEC facts *filed* by $t$. It trades at the close of $t$ and is
   measured against the return from $t$ to $t+h$.
3. **Measures the information coefficient**, the cross-sectional rank
   correlation between the signal and forward market-excess returns, at 1, 5
   and 21 days. It tests four signals: sentiment level, sentiment surprise
   versus the ticker's own 60-day average, sentiment with recent momentum
   removed, and fundamentals. Standard errors are Newey-West, which accounts
   for overlapping return windows.
4. **Calibrates the views** with a pooled regression and Driscoll-Kraay
   standard errors:

   $$\frac{r_{i,t\to t+h}\cdot 252/h}{\sigma_i} = \kappa\, x_{i,t} + \varepsilon$$

   Before use, $\hat\kappa$ is shrunk toward zero with the empirical-Bayes
   (positive-part James-Stein) factor $\tilde\kappa = \hat\kappa\,(1 - 1/t^2)^+$,
   so an estimate with $|t| \le 1$ produces no views and weak ones are scaled
   down rather than traded at face value. The live model then uses
   $Q_i = \pi_i + \sigma_i \tilde\kappa x_i$, with $\Omega$ taken from the
   uncertainty in $\tilde\kappa$ plus the noise in each ticker's average
   headline score.
5. **Runs a walk-forward backtest** of five strategies with monthly rebalancing
   and transaction costs: market cap, equal weight, Black-Litterman without
   views, with the hand-tuned views, and with calibrated views. The calibrated
   strategy re-estimates $\kappa$ at each rebalance using only returns that
   had already been realized. The report includes Sharpe ratio, maximum
   drawdown, turnover, information ratio and bootstrap confidence intervals.

Results go to `evaluation/`. The fitted calibration is installed for
`analyze` and `serve`, which use it automatically (`--no-calibration` opts
out). The dashboard's **Evaluation** tab shows the latest report.

### Results

Run on 6 October 2026: 24 US large caps across sectors, October 2024 to October
2026, 25,724 headlines from the Google News archive (87% of ticker-days had at
least 3 headlines in the trailing week). The full report is in
[`evaluation/evaluation.json`](evaluation/evaluation.json).

![Evaluation tab](docs/evaluation-light.png)

**Headline sentiment did not predict returns.** Across all three sentiment
variants and all horizons, the mean IC is between −0.022 and +0.007 and no
t-statistic exceeds 1.3:

| Signal | IC, 1 day | IC, 5 days | IC, 21 days |
|---|---|---|---|
| Sentiment (level) | −0.001 (t −0.12) | −0.020 (t −1.17) | −0.018 (t −0.65) |
| Sentiment (surprise vs 60-day mean) | +0.007 (t +0.63) | −0.005 (t −0.32) | −0.002 (t −0.11) |
| Sentiment excluding 5-day momentum | −0.006 (t −0.50) | −0.022 (t −1.26) | −0.018 (t −0.68) |
| Fundamentals quality | +0.026 (t +2.31) | +0.040 (t +1.63) | +0.047 (t +0.85) |

The only nominally significant result, fundamentals at one day, is one of
twelve tests. A Bonferroni correction would require $|t| \ge 2.87$, so it is
best read as weak evidence. The calibrated coefficients agree: sentiment
$\hat\kappa = -0.65 \pm 0.40$ ($t = -1.62$, shrunk to $-0.40$) and
fundamentals $\hat\kappa = +0.35 \pm 0.52$ ($t = 0.68$, shrunk to zero).

**No strategy beat market-cap weights by a statistically meaningful margin**
(walk-forward, monthly rebalancing, 10 bps costs):

| Strategy | Return | Volatility | Sharpe | Max drawdown | Turnover | Active vs market cap (95% CI) |
|---|---|---|---|---|---|---|
| Market cap | 26.8% | 15.8% | 1.69 | −18.4% | 10% | — |
| Equal weight | 24.9% | 15.7% | 1.58 | −18.4% | 7% | −1.9% (−5.2% to +0.3%) |
| BL, no views | 26.0% | 15.4% | 1.69 | −18.1% | 10% | −0.8% (−1.5% to +0.1%) |
| BL + hand-tuned views | 24.5% | 16.4% | 1.50 | −18.6% | 57% | −2.3% (−6.5% to +1.8%) |
| BL + calibrated views | 27.1% | 16.3% | 1.66 | −18.1% | 50% | +0.3% (−6.3% to +7.0%) |

What this means:

* **The hand-tuned views cost money.** They added 57% turnover per rebalance
  and trailed market cap by 2.3% a year. That is the outcome calibration
  exists to prevent.
* **Calibration behaved as intended.** With no reliable signal it kept the
  portfolio close to the market prior, and its result is statistically
  indistinguishable from market-cap weights. The first version applied the
  insignificant $\hat\kappa$ at face value and turned over 67% per
  rebalance. That run motivated the shrinkage step, which cut turnover to 50%
  without changing the conclusion.
* **These results are consistent with large, heavily covered stocks pricing
  in news quickly.** Headlines published the day before carry little
  information about the next month's relative returns. Denser or faster
  sources, such as intraday timestamps, smaller companies or full article
  text, are the natural next experiments, and `--headlines-csv` makes them
  pluggable.

## Project layout

```
src/
  data/       prices (Yahoo), sec (EDGAR fundamentals), news (RSS + relevance filter)
  sentiment/  finbert (candle model + tokenizer), qbert (int8 BERT + quantizer), download (pinned,
              verified weights), store (persistent score cache), benchmark (precision comparison)
  model/      covariance (Ledoit-Wolf), black_litterman, signals + calibration (views), optimizer (FISTA)
  research/   archive (historical headlines), panel (point-in-time signals), stats (IC, HAC SEs,
              bootstrap), backtest (walk-forward), evaluate (orchestration + report)
  pipeline.rs orchestration, caching, share rounding
  server.rs   axum API + embedded dashboard (web/index.html)
  main.rs     CLI: analyze · serve · evaluate · score · download-model
```

## Development

```bash
cargo test                                    # unit tests: math, parsing, signals, rounding
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

CI runs all three on every push.

## Limitations

* **The historical news source is approximate.** The Google News archive has
  only day-level timestamps (handled with a one-day lag), and it reflects which
  articles still exist today. The universe is also chosen today, which
  introduces survivorship bias. A CSV import of a curated dataset avoids the
  first two problems.
* **The analysis benchmarks table isn't a backtest.** It compares portfolios
  under the model's own posterior; use `pathos evaluate` for realized,
  out-of-sample performance.
* **US coverage only.** Fundamentals are available only for SEC registrants.
  ETFs and foreign listings fall back to price-only treatment.
* **Not investment advice.**
