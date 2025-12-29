"""
Alternate historical data retriever that relies solely on AlphaVantage's
`TIME_SERIES_DAILY_ADJUSTED` endpoint. This is a drop-in replacement for
`get_info.py` when Yahoo Finance is unavailable.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from trading_funcs import (
    company_symbols,
    download_stock_history_alphavantage,
)
from get_info import (
    update_dynamic_tuning,
    update_info,
)

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def get_historical_info_alphavantage(
    companies: Optional[List[str]] = None,
    api_key: str = ALPHAVANTAGE_API_KEY,
    throttle_seconds: float = 15.0,
) -> None:
    """
    Retrieve historical data for each ticker using AlphaVantage and cache the
    processed indicators under the `Stocks/<ticker>` directory.

    Args:
        companies: Tickers to download. Defaults to `company_symbols`.
        api_key: AlphaVantage API key (env `ALPHAVANTAGE_API_KEY` if set).
        throttle_seconds: Seconds to sleep between API calls to respect the
            documented 5 calls/minute limit.
    """
    if not api_key:
        raise ValueError(
            "AlphaVantage API key is missing. Set ALPHAVANTAGE_API_KEY env var."
        )

    tickers = companies or company_symbols
    for ticker in tickers:
        print(f"[AlphaVantage] Downloading {ticker} ...")
        stock_data = download_stock_history_alphavantage(ticker, api_key=api_key)
        if stock_data.empty:
            raise ConnectionError(
                f"AlphaVantage returned no data for {ticker}. "
                "Confirm the symbol and that the API limit hasn't been exceeded."
            )

        os.makedirs(f"Stocks/{ticker}", exist_ok=True)
        update_dynamic_tuning(ticker, stock_data)
        update_info(ticker, stock_data)

        # Respect AlphaVantage's limit of 5 calls per minute.
        time.sleep(throttle_seconds)


if __name__ == "__main__":
    get_historical_info_alphavantage()
