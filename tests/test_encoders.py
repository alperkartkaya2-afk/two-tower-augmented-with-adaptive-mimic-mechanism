import torch

from src.models.encoders import build_tower_encoder


def test_build_tower_encoder_with_features():
    config = {
        "type": "tower",
        "id_embedding": {"params": {"embedding_dim": 8}},
        "feature_encoder": {"type": "linear", "output_dim": 8},
        "fusion": "gated",
        "adaptive_mimic": {"hidden_dim": 16},
    }
    encoder = build_tower_encoder(
        config,
        num_embeddings=5,
        feature_dim=4,
        device=torch.device("cpu"),
    )

    inputs = {
        "indices": torch.tensor([0, 1, 2], dtype=torch.long),
        "features": torch.randn(3, 4),
    }
    output = encoder(inputs)
    assert output.shape == (3, 8)


def test_build_tower_encoder_sparse_embedding():
    config = {
        "type": "tower",
        "id_embedding": {"params": {"embedding_dim": 4, "sparse": True}},
        "fusion": "identity",
    }
    encoder = build_tower_encoder(
        config,
        num_embeddings=10,
        feature_dim=0,
        device=torch.device("cpu"),
    )

    assert getattr(encoder.embedding, "sparse", False)
