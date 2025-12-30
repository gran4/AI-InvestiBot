"""
Name:
    Models.py

Purpose:
    This module provides the classes for all the models which can be trained and used to 
    predict stock prices. The models themselves all inherit the methods from the BaseModel 
    with variations in symbols and information keys etc.

Author:
    Grant Yul Hur

See also:
    Other modules related to running the stock bot -> lambda_implementation, loop_implementation
"""

import json
import os

from typing import Any, Optional, Union, Callable, List, Dict, Tuple
from warnings import warn
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
try:
    from .custom_objects import (
        DirectionalConsistencyLoss,
        ReversalHuberLoss,
        create_LSTM_model,
        create_LSTM_model2,
        create_lightweight_model,
        create_context_gated_model,
        create_probabilistic_model,
        create_directional_model,
    )
except ImportError:
    from custom_objects import (
        DirectionalConsistencyLoss,
        ReversalHuberLoss,
        create_LSTM_model,
        create_LSTM_model2,
        create_lightweight_model,
        create_context_gated_model,
        create_probabilistic_model,
        create_directional_model,
    )

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from pandas_market_calendars import get_calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from trading_funcs import (
    check_for_holidays, get_relavant_values,
    create_sequences, process_flips,
    non_daily, non_daily_no_use, is_floats,
    calculate_percentage_movement_together,
    indicators_to_add_noise_to, indicators_to_scale,
    scale_indicators,
)
try:
    from .directional_labels import generate_directional_labels
except ImportError:
    from directional_labels import generate_directional_labels
from get_info import (
    calculate_momentum_oscillator,
    get_liquidity_spikes,
    get_earnings_history
)


__all__ = (
    'DirectionalConsistencyLoss',
    'ReversalHuberLoss',
    'BaseModel',
    'PriceModel',
    'PercentageModel',
    'ImpulseMACD_indicators',
    'Reversal_indicators',
    'RSI_indicators',
    'Earnings_indicators',
    'break_out_indicators', 
    'super_trends_indicators'
)


MAX_DATA_END_DATE_STR = os.getenv("MAX_DATA_END_DATE", "2025-11-25")
try:
    MAX_DATA_END_DATE = datetime.strptime(MAX_DATA_END_DATE_STR, "%Y-%m-%d").date()
except ValueError as exc:
    raise ValueError(
        f"Invalid MAX_DATA_END_DATE '{MAX_DATA_END_DATE_STR}'. Use YYYY-MM-DD."
    ) from exc


class BaseModel:
    """
    This is the base class for all the models. It handles the actual training, saving,
    loading, predicting, etc. Setting the `information_keys` allows us to describe what
    the model uses. The information keys themselves are retrieved from a json format
    that was created by getInfo.py.

    Args:
        start_date (str): The start date of the training data
        end_date (str): The end date of the training data
        stock_symbol (str): The stock symbol of the stock you want to train on
        num_days (int): The number of days to use for the LSTM model
        information_keys (List[str]): The information keys that describe what the model uses
    """

    def __init__(self, start_date: Optional[Union[date, str]] = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: str = "AAPL",
                 num_days: Optional[int] = None,
                 information_keys: List[str]=["Close"]) -> None:
        self.TRAIN_RATIO = 0.7
        self.VAL_RATIO = 0.15
        self.TEST_RATIO = max(0.0, 1 - self.TRAIN_RATIO - self.VAL_RATIO)
        if num_days is None:
            with open(f'Stocks/{stock_symbol}/dynamic_tuning.json', 'r') as file:
                num_days = json.load(file)['num_days']

        self._start_date: Optional[str] = None
        self._end_date: Optional[str] = None

        self.stock_symbol = stock_symbol
        self.information_keys = list(dict.fromkeys(information_keys))
        self._data_information_keys = self._compute_data_keys()
        self.num_days = num_days
        self._label_offset = 0.0

        self.update_dates(start_date=start_date, end_date=end_date)

        self.model: Optional[Sequential] = None
        self.scaler_data: Dict[str, Dict[str, float]] = {}
        self._manual_holdout_start: Optional[int] = None
        self._cached_train_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_val_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None

#________For offline predicting____________#
        self.cached: Optional[np.ndarray] = None

        # NOTE: cached_info is a pd.DateFrame online,
        # while it is a Dict offline
        self.cached_info: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None
        self._last_train_evaluation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_val_evaluation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_test_evaluation: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._last_test_predictions: Optional[np.ndarray] = None
        self._feature_clip_bounds: Optional[List[Tuple[float, float]]] = None
        self._balance_reference: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @staticmethod
    def _normalize_date(value: Optional[Union[date, str]]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            datetime.strptime(value, "%Y-%m-%d")
            return value
        raise TypeError("Dates must be provided as datetime.date or YYYY-MM-DD strings")

    @property
    def start_date(self) -> str:
        if self._start_date is None:
            raise ValueError("start_date has not been initialized")
        return self._start_date

    @start_date.setter
    def start_date(self, value: Union[date, str]) -> None:
        normalized = self._normalize_date(value)
        if normalized is None:
            raise ValueError("start_date cannot be None once initialized")
        self._start_date = normalized

    @property
    def end_date(self) -> str:
        if self._end_date is None:
            raise ValueError("end_date has not been initialized")
        return self._end_date

    @end_date.setter
    def end_date(self, value: Union[date, str]) -> None:
        normalized = self._normalize_date(value)
        if normalized is None:
            raise ValueError("end_date cannot be None once initialized")
        end_dt = datetime.strptime(normalized, "%Y-%m-%d").date()
        today = date.today()
        cap_date = min(today, MAX_DATA_END_DATE)
        if end_dt > cap_date:
            print(f"[BaseModel] Clamping end_date from {normalized} to {cap_date} to avoid future leakage.")
            end_dt = cap_date
        self._end_date = end_dt.strftime("%Y-%m-%d")

    def update_dates(
            self, start_date=None,
            end_date=None,
        ):
        if end_date is None:
            end_date = date.today()
            #lower type(end_date) == date turns it into string
        if start_date is None:
            with open(f'Stocks/{self.stock_symbol}/dynamic_tuning.json', 'r') as file:
                relevant_years = json.load(file)['relevant_years']
            start_date = end_date - relativedelta(years=relevant_years)

        start_str = self._normalize_date(start_date)
        end_str = self._normalize_date(end_date)
        if start_str is None or end_str is None:
            raise ValueError("start_date and end_date must resolve to valid dates")

        adj_start, adj_end = check_for_holidays(start_str, end_str)
        self.start_date = adj_start
        self.end_date = adj_end

    def _compute_data_keys(self) -> List[str]:
        ordered: List[str] = []
        ordered.append('Close')
        for key in self.information_keys:
            if key not in ordered:
                ordered.append(key)
        return ordered

    def _get_data_keys(self) -> List[str]:
        # Recompute each time in case information_keys changes dynamically.
        self._data_information_keys = self._compute_data_keys()
        return self._data_information_keys

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        return x_total, y_total

    def _split_sequences(
            self,
            x_total: np.ndarray,
            y_total: np.ndarray,
            train_ratio: Optional[float]=None,
            val_ratio: Optional[float]=None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the sequences chronologically into train, validation, and test sets.
        """
        if x_total.shape[0] != y_total.shape[0]:
            raise ValueError("Features and labels must contain the same number of samples")

        total = x_total.shape[0]
        if total < 3:
            raise ValueError("Need at least three samples to create train/val/test splits")

        if train_ratio is None:
            train_ratio = self.TRAIN_RATIO
        if val_ratio is None:
            val_ratio = self.VAL_RATIO

        if not 0 < train_ratio < 1:
            raise ValueError("`train_ratio` must be between 0 and 1")
        if val_ratio < 0 or train_ratio + val_ratio >= 1:
            raise ValueError("`val_ratio` must be non-negative and leave room for the test split")

        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))

        if val_end >= total:
            val_end = total - 1

        if val_end <= train_end or val_end >= total:
            raise ValueError("Not enough samples to build a validation and test split")

        return (
            x_total[:train_end],
            y_total[:train_end],
            x_total[train_end:val_end],
            y_total[train_end:val_end],
            x_total[val_end:],
            y_total[val_end:],
        )

    def _compute_scaler_stats(
        self, x_train_raw: np.ndarray, information_keys: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        if x_train_raw.ndim < 2:
            raise ValueError("Expected x_train_raw to have at least 2 dimensions (samples, features)")
        flattened = x_train_raw.reshape(-1, x_train_raw.shape[-1])
        info_keys = information_keys or self.information_keys
        for idx, key in enumerate(info_keys):
            feature_values = flattened[:, idx]
            min_val = float(np.min(feature_values))
            max_val = float(np.max(feature_values))
            diff = max(max_val - min_val, 1e-9)
            stats[key] = {'min': min_val, 'diff': diff}
        return stats

    def _scale_feature_block(
        self,
        array: np.ndarray,
        scaler_data: Dict[str, Dict[str, float]],
        information_keys: List[str],
    ) -> np.ndarray:
        if array.size == 0:
            return array
        scaled = np.copy(array)
        for idx, key in enumerate(information_keys):
            stats = scaler_data.get(key)
            if stats is None:
                continue
            diff = stats['diff'] if stats['diff'] != 0 else 1e-9
            scaled[..., idx] = (scaled[..., idx] - stats['min']) / diff
            if key in scale_indicators:
                scaled[..., idx] *= scale_indicators[key]
        return scaled

    def _compute_clip_bounds(
        self, array: np.ndarray, percentiles: Tuple[float, float]
    ) -> List[Tuple[float, float]]:
        flattened = array.reshape(-1, array.shape[-1])
        lower = np.percentile(flattened, percentiles[0], axis=0)
        upper = np.percentile(flattened, percentiles[1], axis=0)
        return list(zip(lower.tolist(), upper.tolist()))

    def _apply_clip_bounds(self, array: np.ndarray) -> np.ndarray:
        if array.size == 0 or not self._feature_clip_bounds:
            return array
        clipped = np.copy(array)
        for idx, (low, high) in enumerate(self._feature_clip_bounds):
            clipped[..., idx] = np.clip(clipped[..., idx], low, high)
        return clipped

    def _oversample_binary_labels(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_flat = y.reshape(-1)
        unique = np.unique(y_flat)
        if unique.size < 2:
            return x, y
        counts = {val: int(np.sum(y_flat == val)) for val in unique}
        max_count = max(counts.values())
        extra_x = []
        extra_y = []
        for val, count in counts.items():
            deficit = max_count - count
            if deficit <= 0 or count == 0:
                continue
            indices = np.where(y_flat == val)[0]
            choices = np.random.choice(indices, size=deficit, replace=True)
            extra_x.append(x[choices])
            extra_y.append(np.full(deficit, val, dtype=y.dtype))
        if not extra_x:
            return x, y
        extra_x = np.concatenate(extra_x, axis=0)
        extra_y = np.concatenate(extra_y, axis=0)
        x_balanced = np.concatenate([x, extra_x], axis=0)
        y_balanced = np.concatenate([y, extra_y], axis=0)
        perm = np.random.permutation(x_balanced.shape[0])
        return x_balanced[perm], y_balanced[perm]

    class _StageConfusionMonitor(tf.keras.callbacks.Callback):
        def __init__(self, x_val: np.ndarray, y_val: np.ndarray):
            super().__init__()
            self.x_val = x_val
            self.y_val = y_val
            self.triggered = False

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.x_val, verbose=0)
            if preds.ndim > 1:
                preds = preds[..., 0]
            y_flat = self.y_val.reshape(-1).astype(int)
            pred_flat = (preds >= 0.5).astype(int).reshape(-1)
            tp = int(np.sum((pred_flat == 1) & (y_flat == 1)))
            tn = int(np.sum((pred_flat == 0) & (y_flat == 0)))
            if tp > 0 and tn > 0:
                if not self.triggered:
                    print(f"[DirectionalModel] confusion reached at epoch {epoch + 1}")
                    self.triggered = True
                self.model.stop_training = True

    class _WeightFlipCallback(tf.keras.callbacks.Callback):
        def __init__(
            self,
            x_watch: np.ndarray,
            y_watch: np.ndarray,
            streak: int,
            lower_bound: float = 0.35,
            upper_bound: float = 0.65,
            stage_name: str = "Stage",
            tn_guard: int = 2,
        ):
            super().__init__()
            self.x_watch = x_watch
            self.y_watch = y_watch
            self.streak = streak
            self.counter = 0
            self.flip = False
            self.missing_class = None
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.stage_name = stage_name
            self.tn_guard = tn_guard
            self.tn_counter = 0

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.x_watch, verbose=0)
            if preds.ndim > 1:
                preds = preds[..., 0]
            y_flat = self.y_watch.reshape(-1).astype(int)
            pred_flat = (preds >= 0.5).astype(int).reshape(-1)
            unique = np.unique(pred_flat)
            mean_pred = float(np.mean(preds))
            imbalance = mean_pred <= self.lower_bound or mean_pred >= self.upper_bound
            tn_count = int(np.sum((pred_flat == 0) & (y_flat == 0)))
            if tn_count == 0:
                self.tn_counter += 1
                self.missing_class = 0
            else:
                self.tn_counter = 0
            if unique.size <= 1 or imbalance:
                self.counter += 1
                if unique.size <= 1:
                    self.missing_class = 1 - int(unique[0])
                else:
                    self.missing_class = 0 if mean_pred >= self.upper_bound else 1
            else:
                self.counter = 0
                self.missing_class = None
            if self.counter >= self.streak or self.tn_counter >= self.tn_guard:
                self.flip = True
                missing = self.missing_class if self.missing_class is not None else -1
                print(
                    f"[{self.stage_name}] Auto-balance triggered at epoch {epoch + 1}: "
                    f"pos_ratio={mean_pred:.2f}, missing_class={missing}, tn={tn_count}"
                )
                self.model.stop_training = True
    def _scale_labels(
        self,
        labels: np.ndarray,
        scaler_data: Dict[str, Dict[str, float]],
        information_keys: List[str],
        scale: bool = True,
    ) -> np.ndarray:
        if labels.size == 0:
            return labels
        if not scale:
            return labels
        label_key = 'Close'
        stats = scaler_data.get(label_key)
        if stats is None:
            min_val = float(np.min(labels))
            diff = float(np.ptp(labels))
            if diff == 0:
                diff = 1e-9
            scaled = (labels - min_val) / diff
        else:
            diff = stats['diff'] if stats['diff'] != 0 else 1e-9
            scaled = (labels - stats['min']) / diff
        if label_key in scale_indicators:
            scaled *= scale_indicators[label_key]
        return scaled

    def _train_stage(
        self,
        information_keys: List[str],
        epochs: int,
        patience: int,
        time_shift: int,
        add_scaling: bool,
        add_noise: bool,
        use_transfer_learning: bool,
        test: bool,
        create_model: Callable,
        reinitialize_model: bool,
        scale_labels: bool = True,
        clip_percentiles: Optional[Tuple[float, float]] = None,
        oversample_binary: bool = False,
        stage_name: Optional[str] = None,
        class_weight: Optional[Dict[int, float]] = None,
        stop_on_confusion: bool = False,
        direction_threshold: Optional[float] = None,
        label_flip_rate: float = 0.0,
        auto_balance: bool = False,
        balance_streak: int = 3,
        balance_multiplier: float = 1.5,
        max_balance_attempts: int = 3,
        balance_bounds: Optional[Tuple[float, float]] = (0.35, 0.65),
        force_balance: bool = False,
        watch_balance_reference: bool = False,
        tn_guard: int = 2,
        shuffle_train: bool = False,
    ) -> None:
        if direction_threshold is not None and hasattr(self, "direction_threshold"):
            self.direction_threshold = direction_threshold
        if force_balance:
            self._balance_reference = None
        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        num_days = self.num_days

        data_keys = self._get_data_keys()

        _, raw_data, _ = get_relavant_values(
            stock_symbol, data_keys, start_date=start_date, end_date=end_date, scale=False
        )
        if 'Close' not in data_keys:
            raise ValueError("'Close' missing from data keys; cannot build labels.")
        close_idx = data_keys.index('Close')
        label_series = raw_data[:, close_idx]
        feature_indices = [data_keys.index(key) for key in information_keys if key in data_keys]
        if not feature_indices:
            raise ValueError("No valid feature indices found for training.")
        feature_data = raw_data[:, feature_indices]

        split = int(len(feature_data))
        arr = feature_data
        threshold = 1e2
        over_threshold_indices = np.where(arr >= threshold)
        under_threshold_indices = np.where(arr <= -threshold)
        all_extreme_indices = np.hstack([over_threshold_indices, under_threshold_indices])
        extreme_values = arr[all_extreme_indices]
        dataset = feature_data
        label_dataset = label_series
        manual_holdout_start = None
        if test:
            manual_holdout_start = max(num_days + 1, int(split * .8))
            dataset = feature_data[:manual_holdout_start]
            label_dataset = label_series[:manual_holdout_start]
            self._manual_holdout_start = manual_holdout_start
        elif reinitialize_model:
            self._manual_holdout_start = None

        x_total_raw, y_total_raw = create_sequences(dataset, num_days, label_series=label_dataset)
        x_total_raw, y_total_raw = self.process_x_y_total(x_total_raw, y_total_raw, num_days, time_shift)
        if time_shift != 0:
            x_total_raw = x_total_raw[:-time_shift]
            y_total_raw = y_total_raw[time_shift:]

        if self._manual_holdout_start is not None:
            total_samples = x_total_raw.shape[0]
            if total_samples < 2:
                raise ValueError("Not enough samples after reserving the hold-out window.")
            train_fraction = self.TRAIN_RATIO / (self.TRAIN_RATIO + self.VAL_RATIO)
            train_len = max(1, int(total_samples * train_fraction))
            if train_len >= total_samples:
                train_len = total_samples - 1
            x_train_raw = x_total_raw[:train_len]
            y_train_raw = y_total_raw[:train_len]
            x_val_raw = x_total_raw[train_len:]
            y_val_raw = y_total_raw[train_len:]
        else:
            x_train_raw, y_train_raw, x_val_raw, y_val_raw, _, _ = self._split_sequences(x_total_raw, y_total_raw)

        def _is_binary_array(arr: np.ndarray) -> bool:
            if arr.size == 0:
                return False
            uniques = np.unique(arr)
            rounded = np.round(uniques, 4)
            return uniques.size <= 2 and np.all(np.isin(rounded, [0.0, 1.0]))

        if force_balance:
            rng = np.random.default_rng()
            if not _is_binary_array(y_train_raw):
                warn(f"[{stage_name or 'Stage'}] force_balance requested but labels are not binary; skipping.")
            else:
                pos_idx = np.where(y_train_raw >= 0.5)[0]
                neg_idx = np.where(y_train_raw < 0.5)[0]
                min_count = min(pos_idx.size, neg_idx.size)
                if min_count == 0:
                    warn(f"[{stage_name or 'Stage'}] force_balance could not find both classes; skipping.")
                else:
                    pos_sel = rng.choice(pos_idx, min_count, replace=False)
                    neg_sel = rng.choice(neg_idx, min_count, replace=False)
                    balance_idx = np.sort(np.concatenate([pos_sel, neg_sel]))
                    x_train_raw = x_train_raw[balance_idx]
                    y_train_raw = y_train_raw[balance_idx]

        if shuffle_train:
            perm = np.random.permutation(x_train_raw.shape[0])
            x_train_raw = x_train_raw[perm]
            y_train_raw = y_train_raw[perm]

        self.scaler_data = self._compute_scaler_stats(x_train_raw, information_keys)
        x_train = self._scale_feature_block(x_train_raw, self.scaler_data, information_keys)
        y_train = self._scale_labels(
            y_train_raw,
            self.scaler_data,
            information_keys,
            scale=scale_labels,
        )
        x_val = self._scale_feature_block(x_val_raw, self.scaler_data, information_keys)
        y_val = self._scale_labels(
            y_val_raw,
            self.scaler_data,
            information_keys,
            scale=scale_labels,
        )

        def _log_binary_distribution(split: str, labels: np.ndarray) -> None:
            if labels.size == 0:
                return
            unique = np.unique(labels)
            if unique.size == 1 and unique[0] in (0, 1):
                pos = int(np.sum(labels))
                neg = labels.size - pos
            elif np.all(np.isin(np.round(unique, 4), [0.0, 1.0])):
                pos = int(np.sum(labels >= 0.5))
                neg = labels.size - pos
            else:
                return
            label = stage_name or "Stage"
            total = labels.size
            print(f"[{label}] {split} labels: total={total} pos={pos} neg={neg} ({pos/total:.2%} pos)")

        _log_binary_distribution("Train", y_train)
        _log_binary_distribution("Val", y_val)

        labels_are_binary = _is_binary_array(y_train)

        if clip_percentiles:
            self._feature_clip_bounds = self._compute_clip_bounds(x_train, clip_percentiles)
            x_train = self._apply_clip_bounds(x_train)
            x_val = self._apply_clip_bounds(x_val)
        else:
            self._feature_clip_bounds = None

        if oversample_binary and labels_are_binary:
            x_train, y_train = self._oversample_binary_labels(x_train, y_train)

        if label_flip_rate > 0 and y_train.size > 0 and labels_are_binary:
            noise_mask = np.random.rand(y_train.shape[0]) < label_flip_rate
            class_means = {
                0: np.mean(y_train[y_train < 0.5]) if np.any(y_train < 0.5) else 0.0,
                1: np.mean(y_train[y_train >= 0.5]) if np.any(y_train >= 0.5) else 1.0,
            }
            averages = np.where(y_train >= 0.5, class_means[1], class_means[0])
            y_train[noise_mask] = averages[noise_mask]

        if force_balance:
            self._balance_reference = (np.copy(x_train), np.copy(y_train))

        # Dynamic class weighting so whichever class is scarce gets boosted.
        if labels_are_binary:
            if class_weight is None:
                class_weight = {0: 1.0, 1: 1.0}
            positives = float(np.sum(y_train))
            negatives = float(y_train.size - positives)
            if positives > 0 and negatives > 0:
                total = positives + negatives
                dyn_weight_0 = total / (2.0 * negatives)
                dyn_weight_1 = total / (2.0 * positives)
                class_weight = {
                    0: class_weight.get(0, 1.0) * dyn_weight_0,
                    1: class_weight.get(1, 1.0) * dyn_weight_1,
                }
        else:
            if class_weight is not None:
                warn(f"Stage '{stage_name or 'Stage'}' provided class_weight but labels are not binary; ignoring weights.")
            class_weight = None
        self._cached_train_eval = (x_train, y_train)
        self._cached_val_eval = (x_val, y_val)

        if reinitialize_model or self.model is None:
            model = create_model(x_train.shape[1:])
            if use_transfer_learning:
                transfer_model = load_model(f"transfer_learning_model")
                for layer_idx, layer in enumerate(model.layers):
                    if layer.name in transfer_model.layers[layer_idx].name:
                        layer.set_weights(transfer_model.layers[layer_idx].get_weights())
        else:
            model = self.model

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks_list = [early_stopping]
        if stop_on_confusion and x_val.size > 0:
            callbacks_list.append(self._StageConfusionMonitor(x_val, y_val))
        balance_callback = None
        if auto_balance and labels_are_binary:
            lower, upper = (balance_bounds or (0.35, 0.65))
            if watch_balance_reference and self._balance_reference is not None:
                x_watch, y_watch = self._balance_reference
            else:
                x_watch = np.concatenate([x_train, x_val], axis=0) if x_val.size else x_train
                y_watch = np.concatenate([y_train, y_val], axis=0) if y_val.size else y_train
            balance_callback = self._WeightFlipCallback(
                x_watch,
                y_watch,
                streak=balance_streak,
                lower_bound=lower,
                upper_bound=upper,
                stage_name=stage_name or "Stage",
                tn_guard=tn_guard,
            )

        divider = max(1, int(x_train.shape[0] / 2))
        if add_scaling:
            indices_cache = [information_keys.index(key) for key in indicators_to_scale if key in information_keys]
            x_total_copy = np.copy(x_train)
            y_total_copy = np.copy(y_train)
            model.fit(
                x_total_copy * 1.1,
                y_total_copy * 1.1,
                validation_data=(x_val, y_val),
                callbacks=callbacks_list,
                batch_size=64,
                epochs=epochs,
                class_weight=class_weight,
            )
            if indices_cache:
                x_total_p1 = np.copy(x_train[:divider])
                y_total_p1 = np.copy(y_train[:divider])
                if x_total_p1.ndim == 3:
                    x_total_p1[:, :, indices_cache] *= 2
                else:
                    x_total_p1[:, indices_cache] *= 2
                y_total_p1 *= 2
                model.fit(
                    x_total_p1,
                    y_total_p1,
                    validation_data=(x_val, y_val),
                    callbacks=callbacks_list,
                    batch_size=64,
                    epochs=epochs,
                    class_weight=class_weight,
                )
        if add_noise:
            x_total_copy = np.copy(x_train)
            indices_cache = [information_keys.index(key) for key in indicators_to_add_noise_to if key in information_keys]
            feature_std = np.std(x_total_copy, axis=(0, 1), keepdims=True)
            feature_std = np.maximum(feature_std, 1e-4)
            noise = np.random.normal(loc=0.0, scale=0.05, size=x_total_copy.shape) * feature_std
            if indices_cache:
                if x_total_copy.ndim == 3:
                    x_total_copy[:, :, indices_cache] += noise[:, :, indices_cache]
                else:
                    x_total_copy[:, indices_cache] += noise[:, indices_cache]
            else:
                x_total_copy += noise
            model.fit(
                x_total_copy,
                y_train,
                validation_data=(x_val, y_val),
                callbacks=callbacks_list,
                batch_size=64,
                epochs=epochs,
                class_weight=class_weight,
            )

        if labels_are_binary:
            if class_weight is None:
                class_weight = {0: 1.0, 1: 1.0}
        else:
            class_weight = None
        balance_attempts = 0
        while True:
            current_callbacks = callbacks_list.copy()
            if balance_callback:
                current_callbacks.append(balance_callback)
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                callbacks=current_callbacks,
                batch_size=64,
                epochs=epochs,
                class_weight=class_weight,
            )
            if balance_callback and balance_callback.flip and balance_attempts < max_balance_attempts:
                balance_attempts += 1
                balance_callback.flip = False
                balance_callback.counter = 0
                majority = balance_callback.missing_class
                if majority is None:
                    majority = 1
                new_weight = class_weight.copy() if class_weight else {0: 1.0, 1: 1.0}
                new_weight[majority] = new_weight.get(majority, 1.0) * balance_multiplier
                class_weight = new_weight
                if y_train.size > 0:
                    mismatch = np.where(y_train != majority)[0]
                    if mismatch.size:
                        take = max(1, int(0.05 * mismatch.size))
                        np.random.shuffle(mismatch)
                        y_train[mismatch[:take]] = majority
                continue
            break
        self.model = model
        self._post_stage_metrics(stage_name, {"train": (x_train, y_train), "val": (x_val, y_val)})


    def _post_stage_metrics(self, stage_name: Optional[str],
                            datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        """Hook for subclasses to log additional metrics after each stage."""
        return

    def train(
        self,
        epochs: int = 1000,
        patience: int = 5,
        time_shift: int = 0,
        add_scaling: bool = False,
        add_noise: bool = True,
        use_transfer_learning: bool = False,
        test: bool = False,
        create_model: Callable = create_LSTM_model,
        curriculum_stages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        warn("If you saved before, use load func instead")
        if time_shift < 0:
            raise ValueError("`time_shift` must be equal of greater than 0")

        if curriculum_stages:
            stages = curriculum_stages
        else:
            stages = [{
                "information_keys": self.information_keys,
                "epochs": epochs,
                "patience": patience,
                "add_scaling": add_scaling,
                "add_noise": add_noise,
                "test": test,
                "create_model": create_model,
            }]

        previous_keys: Optional[List[str]] = None
        for idx, stage in enumerate(stages):
            stage_keys = stage.get("information_keys", self.information_keys)
            stage_epochs = stage.get("epochs", epochs)
            stage_patience = stage.get("patience", patience)
            stage_scaling = stage.get("add_scaling", add_scaling)
            stage_noise = stage.get("add_noise", add_noise)
            stage_test = stage.get("test", idx == len(stages) - 1 and test)
            stage_model = stage.get("create_model", create_model)
            reinitialize = idx == 0 or previous_keys is None or stage_keys != previous_keys
            stage_name = stage.get("name", f"Stage {idx + 1}")
            if hasattr(self, "direction_threshold") and "direction_threshold" in stage:
                self.direction_threshold = stage["direction_threshold"]
            if hasattr(self, "atr_factor") and "atr_factor" in stage:
                self.atr_factor = stage["atr_factor"]
            self._train_stage(
                information_keys=stage_keys,
                epochs=stage_epochs,
                patience=stage_patience,
                time_shift=time_shift,
                add_scaling=stage_scaling,
                add_noise=stage_noise,
                use_transfer_learning=use_transfer_learning,
                test=stage_test,
                create_model=stage_model,
                reinitialize_model=reinitialize,
                scale_labels=stage.get("scale_labels", True),
                clip_percentiles=stage.get("clip_percentiles"),
                oversample_binary=stage.get("oversample_binary", False),
                class_weight=stage.get("class_weight"),
                stop_on_confusion=stage.get("stop_on_confusion", False),
                direction_threshold=stage.get("direction_threshold"),
                label_flip_rate=stage.get("label_flip_rate", 0.0),
                balance_bounds=stage.get("balance_bounds"),
                max_balance_attempts=stage.get("max_balance_attempts", 3),
                force_balance=stage.get("force_balance", False),
                watch_balance_reference=stage.get("watch_balanced", False),
                tn_guard=stage.get("tn_guard", 2),
                shuffle_train=stage.get("shuffle_train", False),
                stage_name=stage_name,
            )
            previous_keys = stage_keys
        final_keys = stages[-1].get("information_keys", self.information_keys)
        self.information_keys = final_keys

    def save(self, transfer_learning: bool=False, name: Optional[str]=None) -> None:
        """
        This method will save the model using the tensorflow save method. It will also save the data
        into the `json` file format.
        """
        if self.model is None:
            raise LookupError("Compile or load model first")
        if name is None:
            name = ''
        name += self.__class__.__name__

        #_________________Save Model______________________#
        if transfer_learning:
            self.model.save(f"transfer_learning_model")
            return
        self.model.save(f"Stocks/{self.stock_symbol}/{name}_model")

        if os.path.exists(f'Stocks/{self.stock_symbol}/min_max_data.json'):
            with open(f"Stocks/{self.stock_symbol}/min_max_data.json", 'r') as file:
                temp = json.load(file)
            if temp:
                self.scaler_data.update({key: value for key, value in temp.items()})
        if not self.scaler_data:
            raise RuntimeError("No scaler_data found. Ensure training captured scaling stats before saving.")

        with open(f"Stocks/{self.stock_symbol}/min_max_data.json", "w") as json_file:
            json.dump(self.scaler_data, json_file)

    @staticmethod
    def is_homogeneous(arr) -> bool:
        """Checks if any of the models indicators are missing"""
        return len(set(arr.dtype for arr in arr.flatten())) == 1

    def test(self, time_shift: int=0, show_graph: bool=False, scale:bool=True,
             title: str="Stock Price Prediction", x_label: str='', y_label: str='Price',
             report_validation: bool=True, plot_probabilities: bool=False
             ) -> Tuple[float, float, float, float, bool]:
        """
        A method for testing purposes. 
        
        Args:
            time_shift (int): The amount of time to shift the data by(in days)
                EX. allows bot to predict 1 month into the future

        Warning:
            It is EXPENSIVE.
        """
        warn("Expensive, for testing purposes")

        if not self.model:
            raise LookupError("Compile or load model first")

        if time_shift < 0:
            raise ValueError("`time_shift` must be equal of greater than 0")

        start_date = self.start_date
        end_date = self.end_date
        stock_symbol = self.stock_symbol
        information_keys = self.information_keys
        data_keys = self._get_data_keys()
        num_days = self.num_days

        #_________________ GET Data______________________#
        if not self.scaler_data:
            raise LookupError("Scaler data missing. Train the model before testing.")

        total_data_dict, raw_data, _ = get_relavant_values( # type: ignore[arg-type]
            stock_symbol, data_keys, self.scaler_data, scale, start_date, end_date
        )
        if 'Close' not in data_keys:
            raise ValueError("'Close' missing from data keys; cannot build labels.")
        close_idx = data_keys.index('Close')
        label_series = raw_data[:, close_idx]
        feature_indices = [data_keys.index(key) for key in information_keys if key in data_keys]
        if not feature_indices:
            raise ValueError("No valid feature indices found for testing.")
        data = raw_data[:, feature_indices]

        #_________________Process Data for LSTM______________________#
        x_total, y_total = create_sequences(data, num_days, label_series=label_series)
        x_total, y_total = self.process_x_y_total(x_total, y_total, num_days, time_shift)
        if self._feature_clip_bounds:
            x_total = self._apply_clip_bounds(x_total)
        label_dates = total_data_dict["Dates"][num_days:]
        if y_total.shape[0] < len(label_dates):
            label_dates = label_dates[:y_total.shape[0]]

        if time_shift != 0:
            x_total = x_total[:-time_shift]
            y_total = y_total[time_shift:]
            label_dates = label_dates[time_shift:]

        label_dates_full = label_dates.copy()

        fallback_holdout = False
        if self._manual_holdout_start is not None:
            if self._cached_train_eval is not None:
                x_train, y_train = self._cached_train_eval
            else:
                feature_shape = x_total.shape[1:]
                x_train = np.empty((0,) + feature_shape)
                y_train = np.empty(0)
            if self._cached_val_eval is not None:
                x_val, y_val = self._cached_val_eval
            else:
                feature_shape = x_total.shape[1:]
                x_val = np.empty((0,) + feature_shape)
                y_val = np.empty(0)
            holdout_sequence_start = max(0, self._manual_holdout_start - num_days)
            holdout_sequence_start = min(holdout_sequence_start, x_total.shape[0])
            x_test = x_total[holdout_sequence_start:]
            y_test = y_total[holdout_sequence_start:]
            label_dates = label_dates[holdout_sequence_start:]
            if x_test.shape[0] == 0:
                warn("Hold-out window collapsed after filtering; recomputing splits.")
                fallback_holdout = True
                label_dates = label_dates_full
        if self._manual_holdout_start is None or fallback_holdout:
            self._manual_holdout_start = None
            x_train, y_train, x_val, y_val, x_test, y_test = self._split_sequences(x_total, y_total)

        train_label_dates = label_dates_full[:x_train.shape[0]]
        val_label_dates = label_dates_full[x_train.shape[0]:x_train.shape[0] + x_val.shape[0]]

        if self._manual_holdout_start is not None:
            test_label_start = holdout_sequence_start
        else:
            test_label_start = x_train.shape[0] + x_val.shape[0]
        test_label_dates = label_dates_full[test_label_start:test_label_start + x_test.shape[0]]

        self._last_train_evaluation = (x_train, y_train)
        self._last_val_evaluation = (x_val, y_val)
        self._last_test_evaluation = (x_test, y_test)

        if self._manual_holdout_start is None:
            if x_val.shape[0] == 0 or x_test.shape[0] == 0:
                raise ValueError("Not enough samples to evaluate validation/test splits. Adjust the training window.")
        else:
            if x_test.shape[0] == 0:
                raise ValueError("Hold-out window is empty. Extend the training range.")

        def evaluate_segment(name: str, x_seg: np.ndarray, y_seg: np.ndarray,
                             segment_label_dates: List[str],
                             print_results: bool=True, return_predictions: bool=False):
            predictions = self.model.predict(x_seg)
            if predictions.ndim > 1 and predictions.shape[-1] > 1:
                predictions = predictions[..., 0]
            threshold = getattr(self, "decision_threshold", None)
            unique_labels = np.unique(y_seg)
            processed_preds = predictions
            if threshold is not None and unique_labels.size <= 2 and np.all(np.isin(np.round(unique_labels), [0, 1])):
                processed_preds = (predictions >= threshold).astype(float)

            directional, spatial = calculate_percentage_movement_together(y_seg, processed_preds)
            if threshold is not None and unique_labels.size <= 2 and np.all(np.isin(np.round(unique_labels), [0, 1])):
                pos_ratio = float(np.mean(processed_preds))
                print(f"[{name.title()}] Positive ratio: {pos_ratio:.2%}")

            rmse = np.sqrt(mean_squared_error(y_seg, processed_preds))
            if len(y_seg) > 1:
                naive_diffs = np.abs(y_seg[1:] - y_seg[:-1])
                mean_abs = np.mean(naive_diffs)
                rmsse = rmse / mean_abs if mean_abs != 0 else np.inf
            else:
                rmsse = np.inf

            # Strong penalty for low-variation predictions (flat curves attempting to
            # coast through reversal regimes). If the predicted standard deviation is
            # less than 20% of the actual series, zero out directional/spatial and
            # triple the error metrics.
            var_threshold = getattr(self, "flat_variation_ratio", 0.05)
            y_std = float(np.std(y_seg))
            if y_std > 1e-8:
                pred_std = float(np.std(processed_preds))
                variance_ratio = pred_std / (y_std + 1e-8)
                if variance_ratio < var_threshold and print_results:
                    print(f"[{name.title()}] Low-variation warning (ratio={variance_ratio:.3f}).")

            if print_results and segment_label_dates:
                segment_dates = (segment_label_dates[0], segment_label_dates[-1])

                print(f"{name.title()} Window ({segment_dates[0]} → {segment_dates[1]}):")
                print(f"Directional Test: {directional}")
                print(f"Spatial Test: {spatial}")
                print(f"RMSE: {rmse}")
                print(f"RMSSE: {rmsse}")
                print()
            if return_predictions:
                return directional, spatial, rmse, rmsse, predictions
            return directional, spatial, rmse, rmsse

        # Evaluate training window (optional overview)
        train_results = None
        if x_train.shape[0] > 0:
            train_results = evaluate_segment(
                "training",
                x_train,
                y_train,
                train_label_dates,
                print_results=report_validation,
                return_predictions=False
            )

        # Evaluate validation (older) window
        if report_validation and x_val.shape[0] > 0:
            _ = evaluate_segment("validation", x_val, y_val, val_label_dates, print_results=True)

        # Evaluate hold-out test window
        test_results = evaluate_segment(
            "test",
            x_test,
            y_test,
            test_label_dates,
            print_results=True,
            return_predictions=True
        )
        directional_test, spatial_test, test_rmse, test_rmsse, test_predictions = test_results
        self._last_test_predictions = test_predictions
    
        print("Homogeneous(Should be True):")
        homogenous = self.is_homogeneous(data)
        print(homogenous)

        if show_graph:
            days_train = label_dates[-y_test.shape[0]:]
            # Plot the actual and predicted prices
            plt.figure(figsize=(18, 6))

            plot_predictions = test_predictions
            if hasattr(plot_predictions, "ndim") and plot_predictions.ndim > 1 and plot_predictions.shape[-1] > 1:
                plot_predictions = plot_predictions[..., 0]
            label_unique = np.unique(y_test)
            threshold = getattr(self, "decision_threshold", 0.5)
            if not plot_probabilities and label_unique.size <= 2 and np.all(np.isin(np.round(label_unique), [0, 1])):
                plot_predictions = (plot_predictions >= threshold).astype(float)

            predicted_test = plt.plot(days_train, plot_predictions, label='Predicted Test')
            actual_test = plt.plot(days_train, y_test, label='Actual Test')

            plt.title(f'{stock_symbol} {title}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)

            import matplotlib.ticker as ticker
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))

            plt.legend(
                [predicted_test[0], actual_test[0]],#[real_data, actual_test[0], actual_train],
                ['Predicted Test', 'Actual Data']#['Real Data', 'Actual Test', 'Actual Train']
            )
            plt.show()
        return directional_test, spatial_test, test_rmse, test_rmsse, homogenous

    def load(self, name: Optional[str]=None):
        """
        This method will load the model using the tensorflow load method.

        Returns:
            BaseModel: The saved model if it was successfully saved
        """
        if not name:
            name = self.__class__.__name__

        self.model = load_model(f"Stocks/{self.stock_symbol}/{name}_model")
        try:
            with open(f"Stocks/{self.stock_symbol}/min_max_data.json", 'r') as file:
                self.scaler_data = json.load(file)
        except FileNotFoundError:
            pass

        # type: ignore[no-any-return]
        return self.model

    def indicators_past_num_days(self, stock_symbol: str, end_date: str,
                                 information_keys: List[str], scaler_data: Dict[str, Dict[str, float]],
                                 cached_info: pd.DataFrame, num_days: int) -> Dict[str, Union[float, str]]:
        """
        This method will return the indicators for the past `num_days` days specified in the
        information keys. It will use the cached information to calculate the indicators
        until the `end_date`.

        Args:
            information_keys (List[str]): tells model the indicators to use
            scaler_data (Dict[str, Dict[str, float]]): used to scale indicators
            cached_info (pd.DataFrame): The cached information
            num_days (int): The number of days to calculate the indicators for
        
        Returns:
            dict: A dictionary containing the indicators for the stock data
                Values will be floats except some expections tht need to be
                processed during run time
        """
        stock_data = {}

        stock_data['Close'] = cached_info['Close'].iloc[-num_days:]

        ema12 = cached_info['Close'].ewm(span=12, adjust=False).mean()
        ema26 = cached_info['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        span = 9
        signal_line = macd.rolling(window=span, min_periods=1).mean().iloc[-num_days:]

        change = cached_info['Close'].diff()
        if '12-day EMA' in information_keys:
            stock_data['12-day EMA'] = ema12.iloc[-num_days:]
        if '26-day EMA' in information_keys:
            stock_data['26-day EMA'] = ema26.iloc[-num_days:]
        if 'MACD' in information_keys:
            stock_data['MACD'] = macd.iloc[-num_days:]
        if 'Signal Line' in information_keys:
            stock_data['Signal Line'] = signal_line
        if 'Histogram' in information_keys:
            histogram = macd - signal_line
            stock_data['Histogram'] = histogram.iloc[-num_days:]
        if '200-day EMA' in information_keys:
            ewm200 = cached_info['Close'].ewm(span=200, adjust=False)
            ema200 = ewm200.mean().iloc[-num_days:]
            stock_data['200-day EMA'] = ema200
        change = cached_info['Close'].diff().iloc[-num_days:]
        if 'Change' in information_keys:
            stock_data['Change'] = change.iloc[-num_days:]
        if 'Momentum' in information_keys:
            momentum = change.rolling(window=10, min_periods=1).sum().iloc[-num_days:]
            stock_data['Momentum'] = momentum
        if 'RSI' in information_keys:
            gain = change.apply(lambda x: x if x > 0 else 0)
            loss = change.apply(lambda x: abs(x) if x < 0 else 0)
            avg_gain = gain.rolling(window=14).mean().iloc[-num_days:]
            avg_loss = loss.rolling(window=14).mean().iloc[-num_days:]
            relative_strength = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + relative_strength))
        if 'TRAMA' in information_keys:
            # TRAMA
            volatility = cached_info['Close'].diff().abs().iloc[-num_days:]
            trama = cached_info['Close'].rolling(window=14).mean().iloc[-num_days:]
            stock_data['TRAMA'] = trama + (volatility * 0.1)
        bollinger_middle = stock_data['Close'].rolling(window=20, min_periods=1).mean()
        std_dev = stock_data['Close'].rolling(window=20, min_periods=1).std()
        if "Bollinger Middle" in information_keys:
            stock_data['Bollinger Middle'] = bollinger_middle
        if "Above Bollinger" in information_keys:
            bollinger_upper = bollinger_middle + (2 * std_dev)
            above_bollinger = np.where(stock_data['Close'] > bollinger_upper, 1, 0)
            stock_data['Above Bollinger'] = pd.Series(above_bollinger)
        if "Bellow Bollinger" in information_keys:
            bollinger_lower = bollinger_middle - (2 * std_dev)
            bellow_bollinger = np.where(stock_data['Close'] < bollinger_lower, 1, 0)
            stock_data['Bellow Bollinger'] = pd.Series(bellow_bollinger)
        if 'gradual-liquidity spike' in information_keys:
            # Reversal
            stock_data['gradual-liquidity spike'] = get_liquidity_spikes(
                cached_info['Volume'], gradual=True
            ).iloc[-num_days:]
        if '3-liquidity spike' in information_keys:
            stock_data['3-liquidity spike'] = get_liquidity_spikes(
                cached_info['Volume'], z_score_threshold=4
            ).iloc[-num_days:]
        if 'momentum_oscillator' in information_keys:
            stock_data['momentum_oscillator'] = calculate_momentum_oscillator(
                cached_info['Close']
            ).iloc[-num_days:]
        if 'ema_flips' in information_keys:
            #_________________12 and 26 day Ema flips______________________#
            stock_data['ema_flips'] = process_flips(ema12[-num_days:], ema26[-num_days:])
            stock_data['ema_flips'] = pd.Series(stock_data['ema_flips'])
        if 'signal_flips' in information_keys:
            stock_data['signal_flips'] = process_flips(macd[-num_days:], signal_line[-num_days:])
            stock_data['signal_flips'] = pd.Series(stock_data['signal_flips'])
        if 'earning diffs' in information_keys:
            #earnings stuffs
            earnings_dates, earnings_diff = get_earnings_history(stock_symbol)
            
            end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            date = end_datetime - relativedelta(days=num_days)

            stock_data['earnings dates'] = []
            stock_data['earning diffs'] = [] # type: ignore[attr]
            low = scaler_data['earning diffs']['min'] # type: ignore[index]
            diff = scaler_data['earning diffs']['diff'] # type: ignore[index]

            for i in range(num_days):
                if not end_date in earnings_dates:
                    stock_data['earning diffs'].append(0)
                    continue
                i = earnings_dates.index(date)
                scaled = (earnings_diff[i]-low) / diff
                stock_data['earning diffs'].append(scaled)
        if not scaler_data:
            # Scale each column manually
            for column in information_keys:
                if column in non_daily:
                    continue
                low = scaler_data[column]['min'] # type: ignore[index]
                diff = scaler_data[column]['diff'] # type: ignore[index]
                column_values = stock_data[column]
                scaled_values = (column_values - low) / diff
                scaled_values = (column_values - low) / diff
                stock_data[column] = scaled_values
        return stock_data

    def update_cached_info_online(self):
        """
        updates `self.cached_info`

        information_keys is so you can update once to get all the info
        look at `loop_implementation` for reference
        """
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        #_________________ GET Data______________________#
        ticker = yf.Ticker(self.stock_symbol)
        cached_info = self.cached_info
        #NOTE: optimize bettween
        if cached_info is None:
            start_datetime = end_datetime - relativedelta(days=self.num_days*4+20)
            if 'ema_200' in self.information_keys:
                start_datetime = start_datetime - relativedelta(days=200)
            cached_info = ticker.history(start=start_datetime, interval="1d")
            if len(cached_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
        else:
            start_datetime = end_datetime - relativedelta(days=1)
            day_info = ticker.history(start=start_datetime, end=self.end_date, interval="1d")
            if len(day_info) == 0: # type: ignore[arg-type]
                raise ConnectionError("Stock data failed to load. Check your internet")
            cached_info = cached_info.drop(cached_info.index[0])
            cached_info = pd.concat((cached_info, day_info))
        return cached_info

    def update_cached_online(self):
        """
        This method updates the cached data using the internet.
        """
        cached = self.indicators_past_num_days(
            self.stock_symbol, self.end_date,
            self.information_keys, self.scaler_data,
            self.cached_info, self.num_days
        )
        cached = [cached[key] for key in self.information_keys if is_floats(cached[key])]
        self.cached = np.transpose(cached)

    def update_cached_offline(self) -> None:
        """This method updates the cached data without using the internet."""
        warn("For Testing")

        end_date = self.end_date
        #_________________ GET Data______________________#
        if not self.cached_info:
            with open(f"Stocks/{self.stock_symbol}/info.json", 'r') as file:
                cached_info = json.load(file)

            if not self.end_date in cached_info['Dates']:
                raise ValueError("end is before or after `Dates` range")
            end_index = cached_info["Dates"].index(self.end_date)
            cached = []
            for key in self.information_keys:
                if key in non_daily_no_use:
                    continue
                cached.append(
                    cached_info[key][end_index-self.num_days:end_index]
                )
            self.cached = np.transpose(cached)
            self.cached_info = cached_info

            if len(self.cached) == 0:
                raise RuntimeError("Stock data failed to load. Reason Unknown")
        if len(self.cached) != 0:
            i_end = self.cached_info["Dates"].index(end_date)
            day_data = [self.cached_info[key][i_end] for key in self.information_keys]

            #delete first day and add new day.
            self.cached = np.concatenate((self.cached[1:], [day_data]))

    def get_info_today(self) -> Optional[np.ndarray]:
        """
        This method will get the information for the stock today and the
        last relevant days to the stock.

        The cached_data is used so less data has to be retrieved from
        yf.finance as it is held to cached or something else.
        
        Returns:
            np.array: The information for the stock today and the
                last relevant days to the stock
        
        Warning:
            It is better to do this in your own code so online and offline are split
        """
        warn('It is better to do this in your own code so online and offline are split')
        end_datetime = datetime.strptime(self.end_date, "%Y-%m-%d")

        start_datetime = end_datetime - relativedelta(days=1)
        nyse = get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_datetime, end_date=end_datetime+relativedelta(days=2))
        if self.end_date not in schedule.index:
            return None

        try:
            if isinstance(self.cached_info, Dict):
                raise ConnectionError("It has already failed to lead")
            self.cached_info = self.update_cached_info_online()
            self.update_cached_online()
        except ConnectionError as error1:
            warn("Stock data failed to download. Check your internet")
            if type(self.cached_info) is pd.DataFrame:
                self.cached_info = None
            try:
                self.update_cached_offline()
            except ValueError as error2:
                print('exception from online prediction: ', error1)
                print('exception from offline prediction: ', error2)
                raise RuntimeError('Neither the online or offline updating of `cached` worked')

        if self.cached is None:
            raise RuntimeError('Neither the online or offline updating of `cached` worked')

        date_object = datetime.strptime(self.start_date, "%Y-%m-%d")
        next_day = date_object + relativedelta(days=1)
        self.start_date = next_day.strftime("%Y-%m-%d")

        date_object = datetime.strptime(self.end_date, "%Y-%m-%d")
        next_day = date_object + relativedelta(days=1)
        self.end_date = next_day.strftime("%Y-%m-%d")

        #NOTE: 'Dates' and 'earnings dates' will never be in information_keys
        self.cached = np.reshape(self.cached, (1, 60, self.cached.shape[1]))
        return self.cached

    def predict(self, info: Optional[np.ndarray] = None) -> np.ndarray:
        """
        This method wraps the model's predict method using `info`.

        Args: 
            info (Optional[np.ndarray]): the information to predict on.
            If None, it will get the info from the last relevant days back.
        
        Returns:
            np.ndarray: the predictions of the model
                The length is determined by how many are put in.
                So, you can predict for time frames or one day
                depending on what you want.
                The length is the days `info` minus `num_days` plus 1

        :Example:
        >>> obj = BaseModel(num_days=5)
        >>> obj = BaseModel(num_days=5)
        >>> obj.num_days
        5
        >>> temp = obj.predict(info = np.array(
                [2, 2],
                [3, 2],
                [4, 1],
                [3, 2],
                [0, 2]
                [7, 0],
                [1, 2],
                [0, 1],
                [2, 2],
                )
            ))
        >>> print(len(temp))
        4
        """
        if info is None:
            info = self.get_info_today()
        if info is None: # basically, if it is still None after get_info_today
            raise RuntimeError(
                "Could not get indicators for today. It may be that `end_date` is beyond today's date"
            )
        if self.model:
            return self.model.predict(info) # typing: ignore[return]
        raise LookupError("Compile or load model first")

class PriceModel(BaseModel):
    """
    This is the model to predict the price. It is not very good. Use the other one

    Args:
        start_date (str): The start date of the training data
        end_date (str): The end date of the training data
        stock_symbol (str): The stock symbol of the stock you want to train on
        num_days (int): The number of days to use for the LSTM model
        information_keys (List[str]): The information keys that describe what the model uses
    """

    def __init__(self, start_date: str = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: Optional[Union[date, str]] = "AAPL",
                 num_days: int = None,
                 information_keys: List[str]=["Close"]) -> None:
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days,
                       information_keys=information_keys
                       )

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        # NOTE: Strip last day for test
        x_total = x_total[:-1]
        y_total = y_total[:-1]
        return x_total, y_total

    def train(self, epochs: int = 1000, patience: int = 5, time_shift: int = 0, add_scaling: bool = True, add_noise: bool = True, use_transfer_learning: bool = False, test: bool = False, create_model: Callable[..., Any] = create_LSTM_model) -> None:
        return super().train(epochs, patience, time_shift, add_scaling, add_noise, use_transfer_learning, test, create_model)

    def profit(self, pred, prev):
        return pred/prev

class DirectionalModel(BaseModel):
    """
    Predicts whether the next price movement is positive or negative.
    The target is the sign of the close price change, so the network only needs
    to learn direction instead of magnitudes.
    """

    def __init__(self, start_date: Optional[Union[date, str]] = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: str = "AAPL",
                 num_days: Optional[int] = None,
                 information_keys: List[str] = ["Close"],
                 direction_threshold: float = 0.0,
                 atr_factor: float = 1.0,
                 forward_horizon: int = 5) -> None:
        super().__init__(start_date=start_date,
                         end_date=end_date,
                         stock_symbol=stock_symbol,
                         num_days=num_days,
                         information_keys=information_keys)
        self.direction_threshold = direction_threshold
        self.atr_factor = atr_factor
        self.forward_horizon = forward_horizon
        self.decision_threshold = 0.5
        self.threshold_bounds: Tuple[float, float] = (0.35, 0.65)

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        x_total, y_total = super().process_x_y_total(x_total, y_total, num_days, time_shift)
        if x_total.shape[0] <= 1:
            raise ValueError("Not enough data to compute directional labels.")
        values = y_total.flatten()
        idx, labels = generate_directional_labels(
            values,
            forward_horizon=getattr(self, "forward_horizon", 5),
            threshold=self.direction_threshold,
            balance=True,
        )
        mask = idx < x_total.shape[0]
        valid_idx = idx[mask]
        filtered_x = x_total[valid_idx]
        filtered_labels = labels[mask]
        return filtered_x, filtered_labels

    def _binary_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        if y_true.size == 0:
            return None
        preds = np.asarray(y_pred).reshape(-1)
        if preds.ndim > 1:
            preds = preds[:, 0]
        threshold = getattr(self, "decision_threshold", 0.5)
        return float(np.mean((preds >= threshold) == (y_true >= 0.5)))

    def _evaluate_cached_accuracy(self, cache: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[float]:
        if cache is None or cache[0].size == 0:
            return None
        x, y = cache
        preds = self.model.predict(x, verbose=0)
        return self._binary_accuracy(y, preds)

    def test(self, time_shift: int=0, show_graph: bool=False, scale:bool=True,
             title: str="Directional Prediction", x_label: str='', y_label: str='Direction',
             report_validation: bool=True, plot_probabilities: bool=False
             ) -> Tuple[float, float, float, float, bool]:
        directional_test, spatial_test, test_rmse, test_rmsse, homogenous = super().test(
            time_shift, show_graph, scale, title, x_label, y_label, report_validation, plot_probabilities=plot_probabilities
        )
        train_acc = self._evaluate_cached_accuracy(self._last_train_evaluation)
        val_acc = self._evaluate_cached_accuracy(self._last_val_evaluation)
        test_acc = None
        if self._last_test_predictions is not None and self._last_test_evaluation is not None:
            _, y_test = self._last_test_evaluation
            test_acc = self._binary_accuracy(y_test, self._last_test_predictions)

        for label, value in (
            ("Training acc", train_acc),
            ("Validation acc", val_acc),
            ("Test acc", test_acc),
        ):
            if value is not None:
                print(f"{label}: {value * 100:.2f}%")

        return directional_test, spatial_test, test_rmse, test_rmsse, homogenous

    def _predict_probs(self, x: np.ndarray) -> np.ndarray:
        probs = self.model.predict(x, verbose=0)
        if probs.ndim > 1:
            probs = probs[..., 0]
        return probs.reshape(-1)

    def _binarize_preds(self, x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        probs = self._predict_probs(x)
        thresh = threshold if threshold is not None else getattr(self, "decision_threshold", 0.5)
        return (probs >= thresh).astype(int)

    def _binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, Dict[str, int]]:
        y_flat = y_true.reshape(-1).astype(int)
        preds = y_pred.reshape(-1).astype(int)
        tp = int(np.sum((preds == 1) & (y_flat == 1)))
        tn = int(np.sum((preds == 0) & (y_flat == 0)))
        fp = int(np.sum((preds == 1) & (y_flat == 0)))
        fn = int(np.sum((preds == 0) & (y_flat == 1)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def _calibrate_threshold(self, probs: np.ndarray, y_true: np.ndarray) -> None:
        y_flat = y_true.reshape(-1)
        if y_flat.size == 0:
            return
        thresholds = np.linspace(0.15, 0.85, 15)
        best_thresh = self.decision_threshold
        best_f1 = -1.0
        for thresh in thresholds:
            preds = (probs >= thresh).astype(int)
            _, _, f1, _ = self._binary_metrics(y_flat, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        lower, upper = getattr(self, "threshold_bounds", (0.35, 0.65))
        ratio = float(np.mean(probs >= best_thresh)) if probs.size else 0.5
        adjusted_thresh = best_thresh

        if ratio < lower:
            target = lower
            adjusted_thresh = float(np.quantile(probs, 1 - target))
        elif ratio > upper:
            target = upper
            adjusted_thresh = float(np.quantile(probs, 1 - target))

        if np.isnan(adjusted_thresh) or not np.isfinite(adjusted_thresh):
            adjusted_thresh = best_thresh

        if best_f1 > 0:
            self.decision_threshold = float(adjusted_thresh)
            print(
                f"[DirectionalModel] Updated decision threshold to {self.decision_threshold:.3f} "
                f"(F1={best_f1:.3f}, pos_ratio={ratio:.2f})"
            )
            return

        # Fallback: force target ratio by quantile if model collapsed to one class.
        target_ratio = 0.5
        quantile = float(np.quantile(probs, 1 - target_ratio)) if probs.size else self.decision_threshold
        if not np.isfinite(quantile):
            quantile = self.decision_threshold
        min_prob = float(np.min(probs)) if probs.size else 0.0
        max_prob = float(np.max(probs)) if probs.size else 1.0
        if np.isclose(quantile, min_prob) and np.isclose(quantile, max_prob):
            quantile = min_prob + 1e-3
        self.decision_threshold = quantile
        print(f"[DirectionalModel] Using fallback threshold {self.decision_threshold:.3f} (F1 flat at {best_f1:.3f})")

    def _post_stage_metrics(self, stage_name: Optional[str],
                            datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        stage_label = stage_name or "Stage"
        print(f"== {stage_label} binary metrics ==")
        calibrated = False
        for split_name, (x, y) in datasets.items():
            if y.size == 0:
                continue
            probs = self._predict_probs(x)
            preds = (probs >= self.decision_threshold).astype(int)
            precision, recall, f1, cm = self._binary_metrics(y, preds)
            print(f"  {split_name.title()}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} cm={cm}")
            if split_name.lower().startswith("val") and not calibrated:
                self._calibrate_threshold(probs, y)
                calibrated = True
        if not calibrated:
            train = datasets.get("train") or datasets.get("Train")
            if train:
                probs = self._predict_probs(train[0])
                self._calibrate_threshold(probs, train[1])

    def train(self,
              epochs: int=1000,
              patience: int=5,
              time_shift: int=0,
              add_scaling: bool=False,
              add_noise: bool=False,
              use_transfer_learning: bool=False,
              test: bool=False,
              create_model: Callable=create_directional_model,
              curriculum_stages: Optional[List[Dict[str, Any]]]=None,
              ) -> None:
        if curriculum_stages:
            for stage in curriculum_stages:
                stage.setdefault("scale_labels", False)
                stage.setdefault("add_noise", False)
        super().train(
            epochs=epochs,
            patience=patience,
            time_shift=time_shift,
            add_scaling=add_scaling,
            add_noise=False,
            use_transfer_learning=use_transfer_learning,
            test=test,
            create_model=create_model,
            curriculum_stages=curriculum_stages,
        )

class PercentageModel(BaseModel):
    """
    Different model that uses min-max scaling on data and accuracy as output. It handles the actual training, saving,
    loading, predicting, etc. Setting the `information_keys` allows us to describe what
    the model uses. The information keys themselves are retrieved from a json format
    that was created by getInfo.py.

    Args:
        start_date (str): The start date of the training data
        end_date (str): The end date of the training data
        stock_symbol (str): The stock symbol of the stock you want to train on
        num_days (int): The number of days to use for the LSTM model
        information_keys (List[str]): The information keys that describe what the model uses
    """

    def __init__(self, start_date: str = None,
                 end_date: Optional[Union[date, str]] = None,
                 stock_symbol: Optional[Union[date, str]] = "AAPL",
                 num_days: int = None,
                 information_keys: List[str]=["Close"],
                 direction_threshold: float = 0.0) -> None:
        if num_days is None:
            num_days = 6
        super().__init__(start_date=start_date,
                       end_date=end_date,
                       stock_symbol=stock_symbol,
                       num_days=num_days,
                       information_keys=information_keys
                       )
        self.cached_cached = None#(For stock caching on 4d data)
        self.direction_threshold = direction_threshold

    def process_x_y_total(self, x_total, y_total, num_days, time_shift):
        # NOTE: Strips 1st day becuase -0 is 0. Look at `y_total[:-1]`
        t = np.copy(y_total)
        y_total = y_total[1:] / y_total[:-1]
        y_total[np.isinf(y_total) | np.isnan(y_total)] = 1.0
        y_total -= 1.0
        y_total *= 150 # amplify percent change to encourage larger predictions without overpowering loss

        arr = y_total
        threshold = 1e2

        over_threshold_indices = np.where(arr >= threshold)

        # Find indices of values under the negative threshold
        under_threshold_indices = np.where(arr <= -threshold)
        all_extreme_indices = np.hstack([over_threshold_indices, under_threshold_indices])

        # Get the values at the extreme indices
        extreme_values = arr[all_extreme_indices]

        #print("All Extreme Indices:", all_extreme_indices)
        #print("Extreme Values:", extreme_values)
        #print(t[449:454])
        #print(arr[449:454])

        # Drop the trailing element because percentage change uses the next label
        if x_total.shape[0] <= 1:
            raise ValueError("Not enough samples to compute percentage change targets.")
        x_total = x_total[:-1]

        if time_shift != 0:
            x_total = x_total[:-time_shift]
            y_total = y_total[time_shift:]

        if x_total.shape[0] < num_days:
            raise ValueError("Not enough sequences to build the requested window depth.")

        num_windows = x_total.shape[0] - num_days + 1
        scaled_data = np.zeros((num_windows, num_days, x_total.shape[1], x_total.shape[2]))

        for i in range(num_windows):
            window = x_total[i : i + num_days]

            # Calculate the high and low close prices for the current window
            high_close = np.max(window, axis=0)
            low_close = np.min(window, axis=0)

            # Avoid division by zero if high_close and low_close are equal
            scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

            # Scale each column using broadcasting
            scaled_window = (window - low_close) / scale_denominator
            scaled_data[i] = scaled_window

        y_total = y_total[num_days-1:]
        if y_total.shape[0] != num_windows:
            y_total = y_total[:num_windows]
        return scaled_data, y_total

    def process_cached(self, cached: Dict) -> np.ndarray:
        num_days = self.num_days
        temp = []
        for key in self.information_keys:
            temp.append(cached[key])
        temp_cached = []
        for i in range(num_days, len(temp[0])):
            indicators = []
            for indicator in temp:
                min_value = indicator[i-num_days:i].min()
                max_value = indicator[i-num_days:i].max()
                # Scale the Series between its high and low values
                temp_scaled = (indicator[i-num_days:i] - min_value) / (max_value - min_value)
                temp_scaled = temp_scaled.fillna(0)
                indicators.append(temp_scaled.tolist())
            indicators = [
                [float(cell) for cell in row] for row in indicators
            ]
            indicators = list(map(list, zip(*indicators)))
            temp_cached.append(indicators)

        temp_cached = np.array(temp_cached)

        if temp_cached.shape[0] < num_days:
            return np.empty((0, num_days, num_days, len(self.information_keys)))

        num_windows = temp_cached.shape[0] - num_days + 1
        scaled_data = np.zeros((num_windows, num_days, num_days, len(self.information_keys)))

        for i in range(num_windows):
            window = temp_cached[i : i + num_days]

            # Calculate the high and low close prices for the current window
            high_close = np.max(window, axis=0)
            low_close = np.min(window, axis=0)

            # Avoid division by zero if high_close and low_close are equal
            scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

            # Scale each column using broadcasting
            scaled_window = (window - low_close) / scale_denominator
            scaled_data[i] = scaled_window
        return scaled_data

    def indicators_past_num_days(self, stock_symbol: str, end_date: str, information_keys: List[str], scaler_data: Dict[str, Dict[str, float]], cached_info: pd.DataFrame, num_days: int) -> Dict[str, Union[float, str]]:
        num_days *= 3
        return super().indicators_past_num_days(stock_symbol, end_date, information_keys, scaler_data, cached_info, num_days)

    def train(
        self,
        epochs: int = 1000,
        patience: int = 5,
        time_shift: int = 0,
        add_noise: bool = True,
        use_transfer_learning: bool = False,
        test: bool = False,
        create_model: Callable[..., Any] = create_LSTM_model2,
        curriculum_stages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if curriculum_stages is None:
            warmup_epochs = max(20, epochs // 5)
            warmup_patience = max(2, patience // 2)
            curriculum_stages = [
                {
                    "name": "Warm-up",
                    "information_keys": self.information_keys,
                    "epochs": warmup_epochs,
                    "patience": warmup_patience,
                    "add_scaling": False,
                    "add_noise": False,
                    "scale_labels": False,
                    "test": False,
                    "create_model": create_model,
                },
                {
                    "name": "Sharpen",
                    "information_keys": self.information_keys,
                    "epochs": epochs,
                    "patience": patience,
                    "add_scaling": False,
                    "add_noise": add_noise,
                    "scale_labels": False,
                    "test": test,
                    "create_model": create_model,
                    "auto_balance": True,
                },
            ]
        return super().train(
            epochs,
            patience,
            time_shift,
            False,
            add_noise,
            use_transfer_learning,
            test,
            create_model,
            curriculum_stages=curriculum_stages,
        )

    def test(self, time_shift: int = 0, show_graph: bool = False) -> None:
        title: str = "Stock Change Prediction"
        x_label: str = ''
        y_label: str = 'Price Change in %'
        return super().test(time_shift, show_graph, False, title, x_label, y_label)

    def update_cached_offline(self) -> None:
        if self.cached_cached is not None:
            self.cached = self.cached_cached
        super().update_cached_offline()
        self.cached_cached = np.copy(self.cached)


        scaled_data = np.zeros((1, self.num_days, self.cached.shape[0], self.cached.shape[1]))

        # Get the data for the current window using the i-window_size approach
        window = self.cached#self.cached[-self.num_days:]

        # Calculate the high and low close prices for the current window
        high_close = np.max(window, axis=0)
        low_close = np.min(window, axis=0)
        # Avoid division by zero if high_close and low_close are equal
        scale_denominator = np.where(high_close == low_close, 1, high_close - low_close)

        # Scale each column using broadcasting
        scaled_window = (window - low_close) / scale_denominator
        # Store the scaled window in the 3D array
        scaled_data[0] = scaled_window
        self.cached = scaled_data

    def profit(self, pred, prev):
        return pred

# for caching for multiple models
def load_models(model_class: BaseModel=PercentageModel, strategys: List[List[str]]=[], names: List[str]=[], company_symbols: List[str]=["AAPL", "GOOG", "AMZN", "META", 'MSFT', 'TSLA', 'V', 'JPM', 'WMT', 'DIS']):
    """
    Loads all models

    model_classes tells the program what models to use
    """
    if len(names) == 0: # no names given when loading, just use base names
        names = [None for i in range(len(strategys))]

    models = []
    total_info_keys = []
    for info_keys in strategys:
        total_info_keys += info_keys

    for company in company_symbols:
        temp = []
        models.append(temp)
        for i in range(len(strategys)):
            model = model_class(stock_symbol=company, information_keys=strategys[i])
            model.num_days = 14
            model.load(name=names[i])
            temp.append(model)
    return models, total_info_keys

ImpulseMACD_indicators = ['Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']
Reversal_indicators = ['gradual-liquidity spike', '3-liquidity spike', 'momentum_oscillator']
Earnings_indicators = ['earnings dates', 'earning diffs', 'Momentum']
RSI_indicators = ['RSI', 'TRAMA']
break_out_indicators = ['Bollinger Middle',
    'Above Bollinger', 'Bellow Bollinger', 'Momentum']
super_trends_indicators = ['supertrend1', 'supertrend2',
    'supertrend3', '200-day EMA', 'kumo_cloud']


def update_transfer_learning(model: BaseModel,
                             companies: List= ["GE", "DIS", "AAPL", "GOOG", "META"],
                             ) -> None:
    """Updates Tranfer Learning Model"""
    model.end_date = date.today()-relativedelta(days=30)
    use = False
    for company in companies:
        model.stock_symbol = company
        model.update_dates()
        model.end_date = date.today()-relativedelta(days=30)
        model.train(use_transfer_learning=use)
        model.save(transfer_learning=True)

        if not use:
            use = True
    model.stock_symbol = "AMZN"
    model.update_dates()
    model.end_date = date.today()-relativedelta(days=30)
    model.train(test=True)
    model.test(show_graph=True)

def get_initial_public_offering_date(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        ipo_date = info.get('start_date')
        return ipo_date
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    modelclass = PercentageModel
    indicators = [ImpulseMACD_indicators, Reversal_indicators, RSI_indicators, break_out_indicators, super_trends_indicators]
    names = ['ImpulseMACD']#, 'Reversal', 'RSI', 'breakout', 'supertrends']
    test_models = []

    for company in ["AAPL"]:
        for i in range(len(indicators)):
            model = modelclass(stock_symbol=company, information_keys=indicators[i])
            ipo_date = get_initial_public_offering_date(company)
            if ipo_date:
                model.start_date = ipo_date
            model.end_date = "2025-11-17"
            model.num_days = 7

            model.train(epochs=200, use_transfer_learning=False, test=True)
            model_name = names[i] if i < len(names) else None
            model.save(name=model_name)
            test_models.append(model)

    for model in test_models:
        model.end_date = "2025-11-17"
        model.test(show_graph=True)
