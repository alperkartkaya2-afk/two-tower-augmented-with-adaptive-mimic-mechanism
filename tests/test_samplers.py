import torch

from src.data.samplers import sample_negative_items


def test_sample_negative_items_excludes_known_items():
    user_indices = torch.tensor([0, 1], dtype=torch.long)
    positives = {0: {1, 2}, 1: {0}}
    negatives = sample_negative_items(
        user_indices,
        num_items=5,
        positives=positives,
        num_negatives=2,
        device=torch.device("cpu"),
    )

    assert negatives.shape == (2, 2)
    assert all(item not in positives[0] for item in negatives[0].tolist())
    assert all(item not in positives[1] for item in negatives[1].tolist())

