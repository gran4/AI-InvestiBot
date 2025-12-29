"""
Utilities to quickly train multiple models with the new walk-forward hold-out
and capture how each configuration generalizes to truly unseen data.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from custom_objects import (
    create_LSTM_model,
    create_LSTM_model2,
    create_lightweight_model,
    create_context_gated_model,
    create_probabilistic_model,
    create_directional_model,
)
from models import PriceModel


MODEL_BUILDERS = {
    "default": create_LSTM_model,
    "conv2d": create_LSTM_model2,
    "lightweight": create_lightweight_model,
    "gated": create_context_gated_model,
    "probabilistic": create_probabilistic_model,
    "directional": create_directional_model,
}


DEFAULT_EXPERIMENTS = [
    {
        "stock_symbol": "AAPL",
        "information_keys": ["Close", "Momentum", "trend_strength", "returns_zscore"],
        "create_model": "gated",
    },
    {
        "stock_symbol": "NVDA",
        "information_keys": ["Close", "returns_zscore", "volatility_14", "earnings_flag"],
        "create_model": "default",
    },
]


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return DEFAULT_EXPERIMENTS
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict):
        return raw.get("experiments", [])
    return raw


def run_holdout_experiment(
    config: Dict[str, Any],
    default_epochs: int = 200,
    default_patience: int = 5,
) -> Dict[str, Any]:
    """
    Train the configured model with `test=True` so the final evaluation uses
    the reserved window.
    """
    model_kwargs = {
        "stock_symbol": config["stock_symbol"],
        "information_keys": config.get("information_keys", ["Close"]),
        "num_days": config.get("num_days"),
        "start_date": config.get("start_date"),
        "end_date": config.get("end_date"),
    }
    model = PriceModel(**{k: v for k, v in model_kwargs.items() if v is not None})

    builder_key = config.get("create_model", "default")
    create_model = MODEL_BUILDERS.get(builder_key, create_lightweight_model)

    epochs = config.get("epochs", default_epochs)
    patience = config.get("patience", default_patience)

    model.train(
        epochs=epochs,
        patience=patience,
        add_scaling=config.get("add_scaling", True),
        add_noise=config.get("add_noise", True),
        test=True,
        create_model=create_model,
    )
    directional, spatial, rmse, rmsse, homogenous = model.test(show_graph=False)
    now = datetime.utcnow().isoformat()
    return {
        "timestamp": now,
        "stock_symbol": config["stock_symbol"],
        "information_keys": model.information_keys,
        "create_model": builder_key,
        "epochs": epochs,
        "directional_test": directional,
        "spatial_test": spatial,
        "rmse": rmse,
        "rmsse": rmsse,
        "homogenous": homogenous,
    }


def run_batch(
    experiments: Iterable[Dict[str, Any]],
    output: Path,
) -> List[Dict[str, Any]]:
    results = []
    for config in experiments:
        try:
            result = run_holdout_experiment(config)
            results.append(result)
        except Exception as exc:  # pragma: no cover
            results.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "stock_symbol": config.get("stock_symbol"),
                    "error": str(exc),
                }
            )
    _ensure_path(output)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    csv_path = output.with_suffix(".csv")
    if results:
        fieldnames: List[str] = []
        for row in results:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    else:
        csv_path.touch()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple hold-out monitoring experiments.")
    parser.add_argument(
        "--config",
        type=str,
        help="JSON file describing experiments. Format: {\"experiments\": [ { ... } ] }.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/holdout_report.json"),
        help="Location where aggregated metrics will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = _load_config(args.config)
    results = run_batch(experiments, args.output)
    print(f"Wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
