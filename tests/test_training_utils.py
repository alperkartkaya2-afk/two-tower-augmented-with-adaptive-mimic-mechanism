import pandas as pd
import pytest

from src.evaluation.metrics import compute_ranking_metrics
from src.pipelines.training import (
    EarlyStoppingController,
    _extract_metric_value,
    _split_train_validation_test,
)


def test_split_train_validation_test_uses_holdout_and_fraction():
    data = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1],
            "item_idx": [0, 1, 2, 3, 4, 5],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
        }
    )

    train, val, test = _split_train_validation_test(
        data,
        train_fraction=0.6,
        test_fraction=None,
        seed=42,
    )

    # One validation point per user (latest timestamp) should be held out.
    assert len(val) == 2
    # Remaining interactions should be split into train/test, preserving order.
    assert len(train) + len(test) == len(data) - len(val)
    assert len(test) == 2  # 40% of remaining (rounded) with RNG seed and 4 samples.


def test_extract_metric_value_from_ranking_metrics():
    metrics = compute_ranking_metrics(
        {0: [1, 2, 3]},
        {0: {2}},
        [1, 2, 3],
    )

    assert _extract_metric_value(metrics, "recall@2") == pytest.approx(1.0)
    assert _extract_metric_value(metrics, "precision@5") is None


def test_early_stopping_controller_stops_after_patience():
    controller = EarlyStoppingController(metric="recall@10", patience=2, min_delta=0.0)

    assert controller.update(0.3, epoch=1) is False
    assert controller.update(0.29, epoch=2) is False
    assert controller.update(0.28, epoch=3) is True  # no improvement for two epochs
