import json

import pytest

from src.evaluation.metrics import compute_ranking_metrics
from src.pipelines.training import (
    TrainingHistory,
    _write_embedding_summary,
    _write_recommendation_report,
)
from src.reporting import save_loss_curves


def _sample_embedding_stats() -> dict:
    return {
        "user_norms": {"mean": 1.0, "std": 0.1, "min": 0.8, "max": 1.2},
        "item_norms": {"mean": 0.9, "std": 0.05, "min": 0.8, "max": 1.0},
        "item_neighbor_overlap": {"category_overlap_mean": 0.4, "category_overlap_std": 0.05, "k": 5},
        "user_alignment": {"cosine_mean": 0.7, "cosine_std": 0.1},
    }


def test_recommendation_report_includes_loss_curve_and_features(tmp_path):
    loss_path = save_loss_curves(
        {
            "Train": [1.0, 0.8, 0.6],
            "Validation": [1.1, 0.9, 0.7],
            "Test": [1.2, 1.0, 0.8],
        },
        output_path=tmp_path / "loss.png",
    )

    metrics = compute_ranking_metrics(
        {0: [1, 2, 3]},
        {0: {2}},
        [1, 3],
    )

    history = TrainingHistory(
        train_loss=[1.0, 0.8, 0.6],
        val_loss=[1.1, 0.9, 0.7],
        test_loss=[1.2, 1.0, 0.8],
        monitored_metric=[0.2, 0.25, 0.3],
    )

    feature_correlations = [
        {"feature": "category:fiction", "pearson_r": 0.5, "p_value": 0.01},
        {"feature": "author:doe", "pearson_r": -0.3, "p_value": 0.05},
    ]

    recommendations = [
        {
            "user_id": "user-1",
            "user_idx": 0,
            "recommendations": [
                {
                    "asin": "B001",
                    "title": "Sample Book",
                    "author": "Author",
                    "categories": ["Fiction"],
                }
            ],
            "category_match": 0.5,
            "author_match": 0.0,
            "history_categories": {"Fiction"},
            "history_authors": set(),
        }
    ]

    report_path = tmp_path / "report.md"
    _write_recommendation_report(
        report_path,
        metrics_summary=metrics,
        embedding_stats=_sample_embedding_stats(),
        recommendations=recommendations,
        loss_plot_path=loss_path,
        history=history,
        monitor_metric="recall@3",
        best_epoch=3,
        feature_correlations=feature_correlations,
    )

    content = report_path.read_text()
    assert "Loss Curves" in content
    assert "![Loss curves" in content
    assert "Feature | Pearson r" in content
    assert "Sample Book" in content


def test_embedding_summary_json_structure(tmp_path):
    summary_path = tmp_path / "embedding.json"
    correlations = [{"feature": "category:fiction", "pearson_r": 0.6, "p_value": 0.02}]
    mimic_stats = {"user": {"gate_mean": 0.4}, "item": {"gate_mean": 0.6}}

    _write_embedding_summary(
        summary_path,
        embedding_stats=_sample_embedding_stats(),
        mimic_stats=mimic_stats,
        feature_correlations=correlations,
        monitor_metric="recall@10",
        best_epoch=4,
    )

    payload = json.loads(summary_path.read_text())
    assert payload["embedding_stats"]["user_norms"]["mean"] == pytest.approx(1.0)
    assert payload["adaptive_mimic"]["user"]["gate_mean"] == pytest.approx(0.4)
    assert payload["feature_correlations"][0]["feature"] == "category:fiction"
