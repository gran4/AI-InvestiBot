# AI-InvestiBot

AI-InvestiBot provides scripts and utilities for experimenting with indicator-heavy LSTM models, walk-forward validation, and automated decision logic across equities. The repository includes helper modules for gathering data, crafting indicators, training models, and connecting them to decision tooling.

## Contents

- [Highlights](#highlights)
- [Roadmap](#roadmap)
- [Quick Start](#quick-start)
- [Training Workflow](#training-workflow)
  - [Data Retrieval and Caching](#data-retrieval-and-caching)
- [Feature Engineering](#feature-engineering)
- [Model Zoo](#model-zoo)
- [Hold-out Monitoring](#hold-out-monitoring)
- [Decision Automation](#decision-automation)
- [Support](#support)
***

## Highlights

- **Indicator-centric cache**: `get_info.py` produces `Stocks/<TICKER>/info.json` with earnings diffs, kumo clouds, supertrends, and other custom indicators used by the LSTM pipelines.
- **Pluggable model builders**: `PriceModel`, `PercentageModel`, and `DirectionalModel` accept `create_model=...` callbacks, so you can prototype new architectures without rewriting the training loop.
- **Walk-forward workflow**: `train(test=True)` hides the boilerplate for hold-out splits and scaler tracking. `model.test()` reuses the configuration for a fast sanity check.
- **Batch experimentation helpers**: `holdout_monitor.py` accepts a JSON sample describing experiments and emits JSON or CSV summaries to compare indicator mixes or model families.
- **Prototype decision stack**: `decision_maker.py` loads multiple saved strategies, refreshes their cached indicators, and feeds a `DecisionTreeClassifier` for a trade vote.
- **Serverless skeleton**: `implementation.py` includes a Lambda-ready loop for anyone who wants to run lightweight decision logic without a dedicated workstation.

## Roadmap
- Hold-out driven accuracy targets — ✅
- Callback-based training API — ✅
- More validation/tests — 🔄
- PercentageModel follow-ups — 🔄

## Quick Start

> ⚠️ Real-time trading remains experimental. Use the tooling for research/backtesting until further notice.

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure secrets**: Copy `secrets_example.config` → `secrets.config`, add API keys (trading, AlphaVantage fallback, etc.).
3. **Cache data**: run `python -m AI-InvestiBot.get_info` with the symbol flag set to `AAPL`. The helper fetches Yahoo Finance history (with AlphaVantage or Stooq fallbacks) and writes `Stocks/<SYMBOL>/info.json`.
4. **Train a model**:
   ```python
   from AI-InvestiBot.models import PriceModel
   from AI-InvestiBot.custom_objects import create_lightweight_model

   model = PriceModel(stock_symbol="AAPL", information_keys=["Close","returns_zscore"])
   model.train(create_model=create_lightweight_model, epochs=300, patience=10, test=True)
   model.save("Stocks/AAPL/MyPriceModel")
   ```
5. **Evaluate hold-out**: 
   ```python
   directional, spatial, rmse, rmsse, homogenous = model.test(show_graph=False)
   ```
6. **Batch comparisons** (optional): run `python -m AI-InvestiBot.holdout_monitor` with your experiment config path and desired output location to produce JSON and CSV summaries.

Keep `Stocks/` out of version control—models and cached data regenerate locally.

## Training Workflow

### Data Retrieval and Caching

- **Download historical OHLCV** — `trading_funcs.download_stock_history`. Retries Yahoo and falls back to AlphaVantage or Stooq when rate-limited.
- **Derive indicators and metadata** — `get_info.py`. Produces `info.json`, `dynamic_tuning.json`, plus scaler references per symbol.
- **Inspect caches** — `validate_inputs.py`. Spot-checks indicator quality, counts samples, and flags missing or invalid entries.

Each indicator drawer writes to JSON so the training loop can load features with `get_relavant_values()` and feed them into the neural network builders.

### Feature Engineering

Key context-aware indicators you can toggle via `information_keys`:

- `returns_zscore`: Rolling 20-day z-score of daily returns.
- `volatility_14`: Two-week realized volatility for regime awareness.
- `trend_strength`: Relative gap between 50/200-day EMAs.
- `ema_spread_10_40`, `volume_surge`, `atr_14`: Derived in `trading_funcs`.
- `earnings_flag`: Binary flag spanning a ±3-day window around earnings.

Non-daily series (earnings dates/diffs) are aligned alongside daily bars for richer context.

### Model Zoo

- **PriceModel**: Predicts price directly after scaling against historical min/max.
- **PercentageModel**: Predicts percentage returns over sliding windows.
- **DirectionalModel**: Optimized for sign accuracy, with curriculum stages, balanced focal loss, and automatic threshold calibration.
- **Custom builders**:
  - `create_lightweight_model`: Single LSTM layer for rapid prototyping.
  - `create_context_gated_model`: Adds a context gate derived from global averages.
  - `create_probabilistic_model`: Predicts mean + log-variance (heteroscedastic loss).
  - `create_directional_model` / `create_directional_model_focal`: Sigmoid outputs backed by balanced focal loss.

The training loop accepts callbacks, so you can plug in any Keras `Model` factory that matches the expected input shape.

### Hold-out Monitoring

`AI-InvestiBot/holdout_monitor.py` packages the walk-forward pipeline into a CLI:

Run `python -m AI-InvestiBot.holdout_monitor` with `configs/experiments.json` and an output file path. The command reads experiments, executes each training run, and writes its findings to JSON and CSV.

- Reads `{ "experiments": [ { "stock_symbol": "...", "information_keys": [...], "create_model": "gated", ... } ] }`
- Runs `PriceModel.train(test=True)` per entry and records directional/spatial/RMSE metrics for the hold-out slice.
- Writes both JSON and CSV summaries for quick spreadsheet analysis.

## Decision Automation

`decision_maker.py` loads multiple trained strategy models (Impulse MACD, breakout, RSI, supertrends, etc.), builds cached indicator windows (online or offline), and feeds predictions into a `DecisionTreeClassifier`.

Features:

- Offline cache reader for disconnected environments (`Stocks/<SYMBOL>/info.json`).
- Automatic re-computation of indicators per 14-day step when running forward in time.
- Optional `TARGET_SYMBOLS` override to focus the evaluation.
- Designed to slot into the `ResourceManager` flows so you can allocate capital based on aggregated votes rather than a single model.


## Support

- **Discord**: https://dsc.gg/ai-investibot/ (custom vanity link)
- **Issues/PRs**: Please open tickets for bugs, docs gaps, or feature proposals. Contributions should avoid checking in regenerated `Stocks/` assets.
***

Thanks for taking a look at AI-InvestiBot. If you build something useful on top of the framework, feel free to share it through an issue or pull request so others can benefit as well.
