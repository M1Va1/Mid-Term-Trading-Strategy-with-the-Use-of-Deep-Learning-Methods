# Mid-Term Trading Strategy with the Use of Deep Learning Methods

Valerii Petrikov 3rd-year coursework project. The pipeline downloads market and fundamental data, engineers 33 cross-sectional features, trains multiple ML models with Optuna hyperparameter optimization, backtests long-short strategies, and evaluates robustness to market noise.

## Architecture

```
  → coursework.ipynb  (main pipeline)
  → data_loader.py  (feature engineering, 33 features)
  → strategy.py     (model training, Optuna HPO, backtesting)
  → robustness.py   (colored-noise stress testing)
  → results/{timestamp}/*.csv
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| LSTM | Deep learning | Sequence model on monthly feature history |
| GRU | Deep learning | Lighter recurrent alternative to LSTM |
| Transformer | Deep learning | Self-attention over feature sequences |
| LSTM (returns-only) | Deep learning | LSTM trained on returns only|
| GRU (returns-only) | Deep learning | GRU trained on returns only|
| GRU (without fundamentals) | Deep learning | GRU variant without fundamental features |
| LSTM (without fundamentals) | Deep learning | LSTM variant without fundamental features |
| Linear Regression | Baseline | Simple linear model |
| Random Forest | Tree ensemble | Scikit-learn random forest |
| XGBoost | Tree ensemble | Gradient-boosted trees |

All neural models support multiple loss functions: MSE, IC (negative Pearson correlation), pairwise ranking loss, and combined MSE + IC.

## Features (33 total)

- **Fundamental (9):** asset growth, book-to-market, dividend yield, EPS streak, ROE, sales-to-price, log market cap, E/P, leverage
- **Momentum / Price (6):** beta, 1m / 6m / 12m / 36m momentum, return volatility
- **Technical (5):** RSI-14, MACD histogram, ATR-14, SMA-20 ratio, SMA-50 ratio
- **Advanced (9):** change in momentum, dollar volume, idiosyncratic vol, Amihud illiquidity, max return, short-term reversal, 52-week high ratio, return skewness, abnormal turnover
- **Temporal / Regime (3):** quarter, January indicator, volatility regime
- **Derived (1):** beta squared

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
```

## Usage

**1. Download data** (once or to refresh):

**2. Run the full pipeline** (training + evaluation):
```bash
jupyter notebook src/coursework.ipynb
```


## Evaluation Metrics

- **IC** (Information Coefficient) — monthly Spearman rank correlation between predictions and realized returns
- **ICIR** — IC / std(IC), measures prediction consistency across months
- **Sharpe ratio**, **Sortino ratio**, **max drawdown**, **annual return** — portfolio-level performance

## Train / Validation / Test Split

| Split | Period | Purpose |
|-------|--------|---------|
| Train | < 2018 | Model fitting |
| Validation | 2018 -- 2020 | Optuna HPO, model selection |
| Test | 2021+ | Out-of-sample evaluation |

## Robustness Testing

Models are stress-tested by injecting colored noise (white, pink, red, blue, violet, grey) into test-period OHLCV prices at 16 sigma levels from 0 to 1. Features are recomputed from noisy prices, and models are re-evaluated to measure performance degradation.

## Output Structure

Results are saved to `results/{timestamp}/`:
- `baseline_*.csv` — portfolio metrics per model
- `best_params_*.csv` — optimal Optuna hyperparameters
- `robustness_model_*.csv` / `robustness_portfolio_*.csv` — performance under noise
- `strategies_custom_*.csv` — custom strategy variants
