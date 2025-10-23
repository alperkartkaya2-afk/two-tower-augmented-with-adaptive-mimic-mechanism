from src.evaluation.metrics import compute_ranking_metrics


def test_compute_ranking_metrics():
    predictions = {
        0: [3, 2, 1],
        1: [4, 5, 6],
    }
    ground_truth = {
        0: {1, 2},
        1: {4},
    }
    metrics = compute_ranking_metrics(predictions, ground_truth, [1, 2, 3])

    assert metrics.recall[1] == 0.5  # user0 miss, user1 hit => (0 + 1)/2
    assert metrics.precision[1] == 0.5
    assert metrics.hit_rate[1] == 0.5
    assert metrics.recall[3] > metrics.recall[1]
