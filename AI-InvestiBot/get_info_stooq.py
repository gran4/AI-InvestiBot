"""
Entry point to generate cached indicator data using the public Stooq API.
Useful when both Yahoo Finance and AlphaVantage are unavailable.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

from trading_funcs import (
    company_symbols,
    download_stock_history_stooq,
)
from get_info import (
    update_dynamic_tuning,
    update_info,
)


def get_historical_info_stooq(
    companies: Optional[List[str]] = None,
    throttle_seconds: float = 2.0,
) -> None:
    tickers = companies or list(company_symbols)
    for ticker in tickers:
        print(f"[Stooq] Downloading {ticker} ...")
        stock_data = download_stock_history_stooq(ticker)
        if stock_data.empty:
            raise ConnectionError(
                f"Stooq returned no data for {ticker}. "
                "Verify the symbol or try again later."
            )

        os.makedirs(f"Stocks/{ticker}", exist_ok=True)
        update_dynamic_tuning(ticker, stock_data)
        update_info(ticker, stock_data)

        # Stooq is fairly permissive, but pause briefly to avoid hammering.
        time.sleep(throttle_seconds)


if __name__ == "__main__":
    get_historical_info_stooq()
