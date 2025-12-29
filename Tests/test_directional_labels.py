import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "AI-InvestiBot" / "directional_labels.py"
SPEC = importlib.util.spec_from_file_location("directional_labels", MODULE_PATH)
directional_labels = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # for type checking
SPEC.loader.exec_module(directional_labels)  # type: ignore[union-attr]
generate_directional_labels = directional_labels.generate_directional_labels


def test_balanced_when_changes_exist():
    values = np.array([100, 110, 90, 120, 80], dtype=float)
    idx, labels = generate_directional_labels(values, forward_horizon=1, threshold=0.0)
    assert idx.size == labels.size
    assert {0.0, 1.0}.issubset(set(labels))


def test_fallback_when_filtered_empty():
    values = np.linspace(100, 100.1, 20, dtype=float)
    idx, labels = generate_directional_labels(values, forward_horizon=5, threshold=1.0)
    assert idx.size == labels.size > 0
    assert labels.dtype == float


def test_order_and_alignment_after_balancing():
    values = np.array([1.0, 1.5, 1.4, 1.9, 1.2, 2.0], dtype=float)
    idx, labels = generate_directional_labels(values, forward_horizon=1, threshold=0.0)
    changes = np.sign(values[1:] - values[:-1])
    assert np.all(labels == (changes[idx] >= 0).astype(float))


def test_balancing_spans_full_range():
    # Positive changes are concentrated near both ends to ensure sampling covers entire window.
    values = np.array(
        [100, 105, 99, 103, 102, 104, 98, 97, 110, 95, 120, 118, 125],
        dtype=float,
    )
    idx, _ = generate_directional_labels(values, forward_horizon=1, threshold=0.0)
    assert idx.min() == 0
    assert idx.max() >= len(values) // 2
