import json
from pathlib import Path
from typing import Dict, List, Optional

from pandas_market_calendars import get_calendar
from models import (
    load_models,
    break_out_indicators,
    ImpulseMACD_indicators,
    Reversal_indicators,
    RSI_indicators,
    super_trends_indicators,
)
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


trading_calendar = get_calendar('XNYS')
_OFFLINE_INFO_CACHE: Dict[str, pd.DataFrame] = {}
_ONLINE_DATA_AVAILABLE = True
TARGET_SYMBOLS: Optional[List[str]] = None  # Provide a list like ["AAPL","MSFT"] to limit output


def _load_offline_info(stock_symbol: str) -> pd.DataFrame:
    cache = _OFFLINE_INFO_CACHE.get(stock_symbol)
    if cache is not None:
        return cache
    info_path = Path("Stocks") / stock_symbol / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Offline cache not found for {stock_symbol}")
    with info_path.open() as f:
        raw = json.load(f)
    if 'Dates' not in raw:
        raise ValueError(f"info.json for {stock_symbol} missing 'Dates'")
    index = pd.to_datetime(raw['Dates'])
    df = pd.DataFrame(index=index)
    length = len(index)
    for key, values in raw.items():
        if key == 'Dates':
            continue
        series = pd.Series(values)
        if series.size < length:
            series = series.reindex(range(length))
        elif series.size > length:
            series = series.iloc[:length]
        df[key] = series.to_numpy()
    _OFFLINE_INFO_CACHE[stock_symbol] = df
    return df


def save_data_for_predictions(company_models, start_date, total_info_keys):
    predictions = [[] for _ in company_models]
    initial_date = pd.Timestamp(start_date, tz='America/Los_Angeles').tz_convert('UTC')
    new_date = trading_calendar.valid_days(
        start_date=initial_date,
        end_date=initial_date + pd.DateOffset(days=14)
    )[-1]
    comparison_date = pd.Timestamp("2023-10-11", tz='America/Los_Angeles').tz_convert('UTC')
    assert new_date.tzinfo == initial_date.tzinfo

    first_model = company_models[0]

    def _offline_slice(end_date_str: str) -> pd.DataFrame:
        offline_df = _load_offline_info(first_model.stock_symbol)
        sliced = offline_df.loc[:end_date_str]
        if sliced.empty:
            raise RuntimeError(
                f"Offline cache for {first_model.stock_symbol} has no data up to {end_date_str}."
            )
        return sliced

    def build_cached_window(end_date: pd.Timestamp):
        """Refresh indicator cache for the provided window end date."""
        # Use the model helpers to gather a fresh batch of indicators for
        # `end_date` without mutating the long-lived model state. Falls back
        # to the recorded info.json data when yfinance is unreachable.
        end_date_str = end_date.tz_convert('America/Los_Angeles').strftime("%Y-%m-%d")
        original_end_date = first_model.end_date
        original_cached_info = first_model.cached_info
        cached_info = None
        try:
            first_model.end_date = end_date_str
            first_model.cached_info = None
            global _ONLINE_DATA_AVAILABLE
            if _ONLINE_DATA_AVAILABLE:
                try:
                    cached_info = first_model.update_cached_info_online()
                except ConnectionError:
                    print("[decision_maker] Failed to reach Yahoo Finance. Falling back to cached data only.")
                    _ONLINE_DATA_AVAILABLE = False
            if cached_info is None:
                cached_info = _offline_slice(end_date_str)
        finally:
            first_model.end_date = original_end_date
            first_model.cached_info = original_cached_info
        if cached_info is None:
            raise RuntimeError("Unable to build cached window")
        cached = first_model.indicators_past_num_days(
            first_model.stock_symbol,
            end_date_str,
            total_info_keys,
            first_model.scaler_data,
            cached_info,
            first_model.num_days
        )
        return cached

    while new_date < comparison_date:
        cached = build_cached_window(new_date)
        for i, model in enumerate(company_models):
            processed_data = model.process_cached(cached)
            if processed_data.size == 0:
                continue
            temp = model.predict(info=processed_data).flatten()
            temp = temp[::-1].tolist()
            predictions[i] += temp
        new_date = trading_calendar.valid_days(
            start_date=new_date,
            end_date=new_date + pd.DateOffset(days=14)
        )[-1]
    return predictions


with open("secrets.config","rb") as f:
    company_information = json.load(f)


def train_decision_tree_classifier(features, labels, test_size=0.2, random_state=42):
    """Train and evaluate a basic decision tree on arbitrary features."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    return clf


def predict_company(decision_tree, feature_vector):
    """Return the predicted class (e.g., company index) for a single feature vector."""
    prediction = decision_tree.predict([feature_vector])[0]
    return prediction


def train_decision_maker(symbols=None):
    # Stored model files include the concrete class name in their prefix
    # (e.g. breakoutPercentageModel_model), so we pass matching identifiers
    # here to ensure load_models picks up the trained weights.
    strategys = [
        break_out_indicators,
        ImpulseMACD_indicators,
        Reversal_indicators,
        RSI_indicators,
    ]
    names = [
        'breakoutPercentageModel',
        'ImpulseMACDPercentageModel',
        'ReversalPercentageModel',
        'RSIPercentageModel',
    ]
    load_kwargs = {
        "strategys": strategys,
        "names": names,
    }
    active_symbols = symbols or TARGET_SYMBOLS
    if active_symbols:
        load_kwargs["company_symbols"] = active_symbols
    models, total_info_keys = load_models(**load_kwargs)
    data = {}
    for company_models in models:
        print(type(company_models[0]))
        data[company_models[0].stock_symbol] = save_data_for_predictions(company_models, "2015-01-01", total_info_keys)
    with open(f"Stocks/data_for_decision_tree.json", "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    train_decision_maker()
