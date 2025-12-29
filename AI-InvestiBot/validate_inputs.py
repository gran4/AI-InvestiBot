"""
Utility script to inspect the cached indicator files (info.json) that feed the
models. It summarizes the numeric indicators, highlights missing/invalid
values, and verifies that each requested key has enough usable samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from trading_funcs import non_daily_no_use

BASE_DIR = Path(__file__).resolve().parent
STOCKS_DIR = BASE_DIR / "Stocks"
if not STOCKS_DIR.exists():
    alt_dir = BASE_DIR.parent / "Stocks"
    if alt_dir.exists():
        STOCKS_DIR = alt_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cached indicator inputs used by the models."
    )
    parser.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="One or more ticker symbols to inspect. Defaults to every directory in Stocks/.",
    )
    parser.add_argument(
        "--info-key",
        action="append",
        dest="info_keys",
        help="Restrict the validation to specific information keys (can be repeated).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=120,
        help="Warn if an indicator has fewer than this many usable samples.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N samples for each numeric indicator for a quick visual spot-check.",
    )
    return parser.parse_args()


def load_info(symbol_path: Path) -> Dict[str, Any]:
    info_path = symbol_path / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json for {symbol_path.name}")
    with info_path.open("r") as fp:
        return json.load(fp)


def numeric_stats(values: Iterable[Any]) -> Tuple[List[float], List[int]]:
    cleaned: List[float] = []
    bad_indices: List[int] = []
    for idx, value in enumerate(values):
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            bad_indices.append(idx)
            continue
        if math.isnan(candidate) or math.isinf(candidate):
            bad_indices.append(idx)
            continue
        cleaned.append(candidate)
    return cleaned, bad_indices


def summarize_indicator(
    key: str,
    values: List[Any],
    min_samples: int,
    preview: int,
) -> List[str]:
    cleaned, bad_indices = numeric_stats(values)

    msg = f"{key}: samples={len(values)}, numeric={len(cleaned)}, invalid={len(bad_indices)}"
    details: List[str] = [msg]

    if len(cleaned) == 0:
        details.append("  WARNING: No numeric samples after cleaning")
        return details
    if len(cleaned) < min_samples:
        details.append(f"  WARNING: Only {len(cleaned)} numeric samples (min required {min_samples})")

    min_val = min(cleaned)
    max_val = max(cleaned)
    mean_val = sum(cleaned) / len(cleaned)
    details.append(f"  range=[{min_val:.5f}, {max_val:.5f}] mean={mean_val:.5f}")

    if preview > 0:
        snippet = ", ".join(f"{val:.5f}" for val in cleaned[:preview])
        details.append(f"  preview: {snippet}")

    if bad_indices:
        details.append(f"  bad indices: {bad_indices[:10]}{'...' if len(bad_indices) > 10 else ''}")
    return details


def validate_symbol(
    symbol: str,
    info: Dict[str, Any],
    info_keys: Optional[List[str]],
    min_samples: int,
    preview: int,
) -> List[str]:
    lines = [f"Symbol: {symbol}"]
    if info_keys is None:
        all_keys = sorted(
            key for key in info.keys()
            if isinstance(info[key], list) and key not in non_daily_no_use
        )
    else:
        all_keys = [key for key in info_keys if key in info]

    if not all_keys:
        lines.append("  No numeric indicators to validate.")
        return lines

    baseline_len = None
    for key in all_keys:
        values = info.get(key, [])
        if not isinstance(values, list):
            lines.append(f"  {key}: skipped (not a list)")
            continue
        details = summarize_indicator(key, values, min_samples, preview)
        lines.extend(f"  {entry}" for entry in details)

        # Track mismatched sequence lengths (these cause ragged tensors later).
        if baseline_len is None:
            baseline_len = len(values)
        elif len(values) != baseline_len:
            lines.append(
                f"  WARNING: {key} length {len(values)} differs from baseline {baseline_len}"
            )
    return lines


def discover_symbols(explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    symbols: List[str] = []
    if not STOCKS_DIR.exists():
        return symbols
    for entry in STOCKS_DIR.iterdir():
        if entry.is_dir() and (entry / "info.json").exists():
            symbols.append(entry.name)
    return sorted(symbols)


def main() -> None:
    args = parse_args()
    symbols = discover_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols found. Ensure Stocks/ contains your cached data.")

    for symbol in symbols:
        symbol_path = STOCKS_DIR / symbol
        try:
            info = load_info(symbol_path)
        except FileNotFoundError as exc:
            print(f"{symbol_path.name}: {exc}")
            continue
        lines = validate_symbol(symbol, info, args.info_keys, args.min_samples, args.preview)
        print("\n".join(lines))
        print("-" * 40)


if __name__ == "__main__":
    main()
