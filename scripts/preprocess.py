"""Entry point for dataset preprocessing routines."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from loguru import logger

from src.data import build_training_dataset, load_dataset
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    data_root = Path(data_cfg.get("root", "data"))
    books_file = data_cfg.get("books_file")
    users_file = data_cfg.get("users_file")
    books_limit = data_cfg.get("books_limit")
    interactions_limit = data_cfg.get("interactions_limit")
    min_user_interactions = int(data_cfg.get("min_user_interactions", 0))
    min_item_interactions = int(data_cfg.get("min_item_interactions", 0))

    logger.info("Loading raw data from {}", data_root)
    dataset = load_dataset(
        data_root,
        books_file=books_file,
        interactions_file=users_file,
        books_limit=books_limit,
        interactions_limit=interactions_limit,
    )

    logger.info("Preparing training artifacts for offline storage")
    training_dataset = build_training_dataset(
        dataset,
        stage="train",
        feature_config=data_cfg.get("feature_params", {}),
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    logger.warning(
        "Preprocessing pipeline not yet implemented. "
        "Serialize `training_dataset` to disk when ready."
    )
    logger.debug(
        "Counts | users={} items={} interactions={} | feature_dim(item={}, user={})",
        len(training_dataset.users),
        len(training_dataset.items),
        len(training_dataset.interactions),
        training_dataset.item_feature_matrix.shape[1],
        training_dataset.user_feature_matrix.shape[1],
    )


if __name__ == "__main__":
    main()
