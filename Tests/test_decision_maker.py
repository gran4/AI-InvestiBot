import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "AI-InvestiBot"))
MODULE_PATH = REPO_ROOT / "AI-InvestiBot" / "decision_maker.py"
SPEC = importlib.util.spec_from_file_location("decision_maker", MODULE_PATH)
decision_maker = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # for type checking
SPEC.loader.exec_module(decision_maker)  # type: ignore[union-attr]
save_data_for_predictions = decision_maker.save_data_for_predictions


class _FakeCalendar:
    def valid_days(self, start_date, end_date):
        # Return only the requested end date so the caller advances 14 days each loop.
        return pd.DatetimeIndex([end_date])


class _FakeModel:
    def __init__(self, windows):
        self.stock_symbol = "TEST"
        self.scaler_data = {}
        self.num_days = 1
        self.end_date = "2023-09-01"
        self.cached_info = None
        self._windows = windows
        self.window_log = []

    def update_cached_info_online(self):
        self.window_log.append(self.end_date)
        if self.end_date not in self._windows:
            raise KeyError(f"No window prepared for {self.end_date}")
        return self._windows[self.end_date]

    def indicators_past_num_days(self, _symbol, _end_date, _keys, _scaler, cached_info, _num_days):
        # Pass the cached info straight through so we can control downstream output.
        return {"window": cached_info}

    def process_cached(self, cached):
        value = cached["window"]
        return np.array([[value]], dtype=float)

    def predict(self, info):
        # Reuse the processed window so save_data_for_predictions can append a scalar.
        return info


def test_save_data_advances_cached_windows(monkeypatch):
    monkeypatch.setattr(decision_maker, "trading_calendar", _FakeCalendar())
    windows = {
        "2023-09-15": 1.0,
        "2023-09-29": 2.0,
    }
    model = _FakeModel(windows)
    result = save_data_for_predictions([model], "2023-09-01", total_info_keys=[])
    assert result == [[1.0, 2.0]]
    assert model.window_log == ["2023-09-15", "2023-09-29"]
