import torch

from src.models import AdaptiveMimicMechanism


def test_adaptive_mimic_mechanism_shapes_and_losses():
    mechanism = AdaptiveMimicMechanism(
        num_users=4,
        num_items=6,
        embedding_dim=8,
        init_std=0.01,
    )
    user_indices = torch.tensor([0, 1], dtype=torch.long)
    item_indices = torch.tensor([2, 3], dtype=torch.long)
    user_embeddings = torch.zeros((2, 8), dtype=torch.float32)
    item_embeddings = torch.ones((2, 8), dtype=torch.float32)

    augmented_user, augmented_item, loss_u, loss_i = mechanism(
        user_indices=user_indices,
        item_indices=item_indices,
        user_embedding=user_embeddings,
        item_embedding=item_embeddings,
    )

    assert augmented_user.shape == user_embeddings.shape
    assert augmented_item.shape == item_embeddings.shape
    assert torch.is_tensor(loss_u) and loss_u.item() >= 0
    assert torch.is_tensor(loss_i) and loss_i.item() >= 0

    neg_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    neg_embeddings = torch.randn((4, 8))
    augmented_neg = mechanism.augment_items(neg_indices, neg_embeddings)
    assert augmented_neg.shape == neg_embeddings.shape
