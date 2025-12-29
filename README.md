# AI-InvestiBot

An extensible research sandbox for experimenting with indicator-rich LSTM architectures, walk-forward validation, and automated decision making across multiple equities. The repo bundles a data pipeline, a suite of model builders, and utilities for benchmarking and deployment.

## Contents

- [Highlights](#highlights)
- [Roadmap](#roadmap)
- [Quick Start](#quick-start)
- [Training Workflow](#training-workflow)
  - [Data Retrieval & Caching](#data-retrieval--caching)
  - [Feature Engineering](#feature-engineering)
  - [Model Zoo](#model-zoo)
  - [Hold-out Monitoring](#hold-out-monitoring)
- [Decision Automation](#decision-automation)
- [Results Snapshot](#results-snapshot)
- [Support](#support)

---

## Highlights

- **Indicator-first mindset**: `get_info.py` builds a consistent JSON cache per ticker with bespoke indicators (earnings deltas, supertrends, kumo clouds, etc.) plus any custom keys you specify.
- **Flexible model creation**: `PriceModel`, `PercentageModel`, and `DirectionalModel` accept callbacks via `create_model=...`, letting you swap between LSTM variants (`create_lightweight_model`, `create_context_gated_model`, probabilistic heads, etc.) without touching the training loop.
- **Walk-forward discipline**: `train(test=True)` automatically reserves a hold-out window and captures scaler statistics from the train split only. `model.test()` mirrors those settings to produce realistic directional/spatial/RMSE metrics.
- **Batch experimentation**: `AI-InvestiBot/holdout_monitor.py` consumes JSON experiment specs and emits JSON/CSV summaries, so you can compare architectures/indicator mixes with one command.
- **Decision automation**: `decision_maker.py` stitches together multiple trained strategies, tracks cached windows offline, and feeds a `DecisionTreeClassifier` to vote on trades.
- **Serverless-friendly**: Core loops are decoupled from the UI, with a Lambda-ready implementation for those who want to deploy without a 24/7 workstation.

## Roadmap

| Goal | Status |
| --- | --- |
| ≥80 % accuracy on unseen data | ✅ |
| Callback-driven training API | ✅ |
| Documentation & robustness parity with a production library | 🔄 |
| Finish PercentageModel refactor follow-ups | 🔄 |

Additional focus areas: richer validation (unit tests already cover label builders + decision maker), more examples, and tightening the real-time trading path once the research foundation is stable.

## Quick Start

> ⚠️ Real-time trading remains experimental. Use the tooling for research/backtesting until further notice.

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure secrets**: Copy `secrets_example.config` → `secrets.config`, add API keys (trading, AlphaVantage fallback, etc.).
3. **Cache data**:  
   ```bash
   python -m AI-InvestiBot.get_info --symbol AAPL
   ```
   The helper fetches Yahoo Finance history (with AlphaVantage/Stooq fallbacks) and writes `Stocks/<SYMBOL>/info.json`.
4. **Train a model**:
   ```python
   from AI-InvestiBot.models import PriceModel
   from AI-InvestiBot.custom_objects import create_lightweight_model

   model = PriceModel(stock_symbol="AAPL", information_keys=["Close","returns_zscore"])
   model.train(create_model=create_lightweight_model, epochs=300, patience=10, test=True)
   model.save("Stocks/AAPL/MyPriceModel")
   ```
5. **Evaluate hold-out**: `directional, spatial, rmse, rmsse, homogenous = model.test(show_graph=False)`
6. **Batch comparisons** (optional):
   ```bash
   python -m AI-InvestiBot.holdout_monitor --config configs/experiments.json --output logs/holdout_report.json
   ```

Keep `Stocks/` out of version control—models and cached data regenerate locally.

## Training Workflow

### Data Retrieval & Caching

| Step | Script | Notes |
| --- | --- | --- |
| Download historical OHLCV | `trading_funcs.download_stock_history` | Retries Yahoo; falls back to AlphaVantage/Stooq if rate-limited. |
| Derive indicators & metadata | `get_info.py` | Outputs `info.json`, `dynamic_tuning.json`, plus scaler references for each symbol. |
| Inspect caches | `validate_inputs.py` | Spot-checks indicator quality, counts samples, and reports missing/invalid entries. |

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

```bash
python -m AI-InvestiBot.holdout_monitor \
  --config configs/experiments.json \
  --output logs/holdout_report.json
```

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

## Results Snapshot

Directional accuracy pulled from representative hold-out runs (AAPL dataset):

| Model | Directional | Spatial | RMSE | RMSSE |
| --- | --- | --- | --- | --- |
| Day Trade | 97.89 % | 95.07 % | 1.34 | 24.99 |
| Impulse MACD | 96.48 % | 95.07 % | 0.69 | 7.99 |
| Reversal | 97.18 % | 95.07 % | 1.13 | 24.43 |
| Earnings | 98.59 % | 96.48 % | 0.87 | 15.58 |
| RSI | 97.14 % | 95.71 % | 0.58 | 22.23 |
| Breakout | 97.89 % | 93.66 % | 1.09 | 21.42 |
| Super Trends | 97.89 % | 92.25 % | 1.69 | 78.60 |

*Numbers vary depending on indicator mix, date window, and architecture. Always re-run experiments with your own configuration before drawing conclusions. `model.test(show_graph=True)` outputs comparison plots to visually inspect alignment.*

### Interpreting the Metrics

- **Directional**: Share of samples where prediction and ground truth moved the same direction (a hit rate).
- **Spatial**: Ensures predictions stay on the correct side of the target after a move, penalizing phase errors.
- **RMSE/RMSSE**: Root mean squared error and its scaled counterpart; lower is better. RMSSE weights large misses more heavily.
- **Confidence**: Hold-out splits guarantee the reported metrics come from unseen data. Early stopping curbs overfitting, and transfer learning is disabled unless you explicitly enable it.

## Support

- **Discord**: https://dsc.gg/ai-investibot/ (custom vanity link)
- **Issues/PRs**: Please open tickets for bugs, docs gaps, or feature proposals. Contributions should avoid checking in regenerated `Stocks/` assets.

---

Thanks for exploring AI-InvestiBot. The repo remains under active development—expect frequent refactors as the research backlog turns into production-ready components. If you build something on top of the framework, let us know! We’re keen to highlight community workflows in future docs.
