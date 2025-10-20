# Architecture Overview

## Objectives
- Support a modular two-tower recommender architecture with minimal coupling.
- Keep configuration-driven workflows to simplify experimentation and deployment.
- Allow gradual introduction of the adaptive mimic mechanism without disruptive refactors.

## Component Breakdown
1. **Data Layer (`src/data/`)**
   - `loaders.py` centralises CSV ingestion and schema validation.
     It defaults to `books.csv` / `users.csv` and gracefully falls back to the trimmed samples when the full dumps are absent.
   - `indexers.py` maintains reversible ID-index mappings used by embedding layers.
   - `preprocessing.py` materialises indexed user/item tables, engineered feature matrices (numeric, categorical, text statistics), per-user positives, and prunes low-activity users/items based on configurable thresholds.
   - `datasets.py` wraps pandas interactions for PyTorch dataloaders.
   - `samplers.py` encapsulates negative sampling strategies.
2. **Model Layer (`src/models/`)**
   - `two_tower.py` orchestrates user and item encoders and exposes similarity scoring.
   - `encoders.py` now builds ID embeddings, feature projection MLPs (default 192→96) and an adaptive mimic fusion module for metadata-aware towers.
3. **Pipeline Layer (`src/pipelines/`)**
   - `training.py` orchestrates data splits, optimisation with sampled negatives, validation metrics, embedding diagnostics, and recommendation audits with Markdown reporting.
   - Future modules can extend evaluation reporting and online inference adapters.
4. **Utilities (`src/utils/`)**
   - `config.py` reads YAML configs; additional helpers can grow alongside experiments.
5. **Interfaces (`scripts/`)**
   - CLI entry points bridge configurations and pipelines for reproducible workflows.

## Data Flow (Current State)
1. CLI scripts parse configuration and resolve dataset locations.
2. Indexed datasets are built with reversible ID mappings, low-frequency users/items are filtered out via configurable thresholds, and per-user interaction sets are captured.
3. Interactions are split via per-user hold-out, batched into PyTorch loaders, and augmented with on-the-fly negative sampling.
4. The two-tower embeddings train under a sampled BCE objective with hybrid dense/sparse optimisers (SparseAdam for giant ID tables), then compute ranking metrics, run embedding diagnostics, and emit a recommendation report with top-K previews.

## Next Additions
- Richer user/item encoders (text features, metadata towers, adaptive mimic modules).
- Feature engineering pipelines (alternative negative sampling, temporal decay, caching).
- Training utilities (checkpointing, early stopping, hyperparameter sweeps).
- Experiment tracking integration (e.g., MLflow, Weights & Biases).
