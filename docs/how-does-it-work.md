# How Does It Work?

This guide explains the project end-to-end: where the data comes from, how it is transformed, what the two‑tower model does, how we train it, and how we evaluate/interpret the results. Treat it as a living manual for engineers, data scientists, and product stakeholders.

> **In a hurry?**
> - Data → indexed & filtered → feature tensors  
> - Towers → ID embeddings + metadata encoders with adaptive‑mimic fusion  
> - Training → sampled BCE with hybrid optimizers  
> - Evaluation → ranking metrics, embedding diagnostics, feature correlations, qualitative audits → Markdown report

---

## 1) System Overview

```mermaid
flowchart LR
    A[Raw CSVs<br/>books.csv & users.csv] --> B[Loaders<br/>(limits + schema checks)]
    B --> C[Preprocessing<br/>filters + feature matrices]
    C --> D[Training Dataset]
    D --> E[Tower Builders<br/>ID + metadata encoders]
    E --> F[Training Loop<br/>negative sampling + BCE]
    F --> G[Evaluation & Diagnostics]
    G --> H[Report<br/>artifacts/reports/*.md]
```

Each step below includes **tech details** and **why it matters** callouts.

---

## 2) Data Layer

| Component | What it does | Why it matters |
|---|---|---|
| **Raw CSVs** (`data/books.csv`, `data/users.csv`) | Amazon book metadata + user interactions. | Not checked into Git; everyone should provide them locally. Trimmed versions exist for fast iteration. |
| **Configuration** (`configs/default.yaml:data`) | File paths, row limits (`books_limit`, `interactions_limit`), minimum interaction thresholds, feature parameters. | Single source of truth for experiments; clone YAMLs for different scenarios. |
| **Loaders** (`src/data/loaders.py`) | Reads CSVs, applies limits, and drops interactions that reference missing books, with logging. | Prevents orphaned interactions and maintains data integrity. |
| **Preprocessing** (`src/data/preprocessing.py`) | Cleans interactions; iteratively enforces min user/item frequency; builds index maps + feature matrices; returns a `TrainingDataset`. | Prunes noisy (ultra‑sparse) users/items, improving both quality and compute cost. |
| **Feature Engineering** (`src/data/features.py`) | Numeric z‑scores; title word/character stats; category & author signals; user aggregates. | Surfaces metadata signals to the model; crucial for cold‑start. |
| **Helpers** (`indexers`, `datasets`, `samplers`) | Index maps, PyTorch `Dataset`, negative sampling. | Backbone for the training/evaluation loops. |

> **Why this design?**  
> Instead of blindly loading all data to GPU/CPU, we sample and filter early to get feedback in minutes—even on the first runs.

---

## 3) Model Layer

### Two‑Tower Architecture (`src/models`)

| Piece | Technical summary | Why it matters |
|---|---|---|
| **ID Embeddings** (`build_id_embedding`) | 96‑dim tables for users/items; `sparse=True` supported. | Scales easily; memory‑friendly with `SparseAdam`. |
| **Feature Encoders** (`build_feature_encoder`) | Encodes metadata vectors via MLP 192 → 96 (optional dropout). | Balances collaborative signals with metadata. |
| **Adaptive Mimic** (`AdaptiveMimicModule`) | Fuses ID + metadata embeddings via a learned gate. | Especially strong for cold‑start users/items. |
| **Tower Wrapper** (`TowerEncoder`) | Fusion mode (`identity`/`sum`/`concat`/`adaptive_mimic`) + device management. | Lets you swap tower fusion via config—not code edits. |
| **TwoTowerModel** | Couples both towers and returns similarity (cosine or dot). | Train/eval code stays agnostic to tower internals. |

> **Technical note:** With sparse embeddings, you can use `padding_idx`, but **do not** combine `sparse=True` with `max_norm`—PyTorch disallows that. Configuration guards against illegal combos.

---

## 4) Training & Evaluation Pipeline

### 4.1 Training (`src/pipelines/training.py`)

1. **Seed & Config Parse** — set `experiment.seed` for reproducibility.  
2. **Dataset Build** — apply min‑interaction thresholds; compute feature matrices; log counts.  
3. **Train/Validation Split** — hold out the latest interaction per user (warns and skips if no timestamps).  
4. **Tower Construction** — config → builders → `TowerEncoder` (log dimensions).  
5. **Optimizer Setup** — Adam/AdamW/SGD for dense params; `SparseAdam` for sparse embeddings.  
6. **Mini‑batch Loop**  
   - Negative sampling (`sample_negative_items`)  
   - User/item embeddings; positive & negative logits  
   - BCE loss → backward → optimizer step  
7. **Epoch Logs** — loss, dataset sizes, tower dims.

### 4.2 Evaluation & Diagnostics

| Step | What it does |
|---|---|
| **Ranking Metrics** | `_evaluate_model` uses a limited candidate pool (positives + random negatives) to compute Recall/Precision/NDCG/MAP. |
| **Embedding Diagnostics** | Norm distributions; nearest‑neighbor category cohesion; user embedding ↔ metadata alignment. |
| **Feature Correlations** | Pearson **r** + **p**‑value between scores and metadata; top informative features reported. |
| **Qualitative Recommendations** | Example top‑K recommendations per user with category/author hit ratios. |
| **Markdown Report** | Everything above lands in `artifacts/reports/recommendation_report.md`. |

> **Tip:** Because evaluation uses sampled negatives, increasing `evaluation.candidate_samples` improves fidelity (at higher cost).

---

## 5) Configuration Cheat Sheet

```yaml
data:
  books_limit: null                # null/None → load all
  interactions_limit: null
  min_user_interactions: 5
  min_item_interactions: 10
  feature_params:
    numeric_columns: ["average_rating", "price", "rating_number"]
    category_top_k: 300
    author_top_k: 300
    user_aggregation: "mean"

model:
  user_encoder:
    id_embedding:
      params: { embedding_dim: 96, sparse: true }
    feature_encoder:
      type: mlp
      hidden_dims: [192]
      output_dim: 96
      dropout: 0.1
    fusion: adaptive_mimic
  item_encoder: { ... }            # mirrors user tower
  similarity: "cosine"
  device: "cpu"

training:
  batch_size: 512
  num_epochs: 10
  negatives_per_positive: 5

evaluation:
  metrics_k: [5, 10, 20]
  candidate_samples: 500

diagnostics:
  item_sample_size: 50
  user_sample_size: 500
  neighbor_k: 5
  feature_corr_sample_size: 20000
  feature_corr_top_k: 20
  report_path: "artifacts/reports/recommendation_report.md"
```

---

## 6) Practical Tips & Caveats

| Scenario | What to do | Why |
|---|---|---|
| **Very large CSVs** | Start with `books_limit` / `interactions_limit`. | Avoids OOM and speeds up iteration. |
| **Over‑filtering leaves too little data** | Lower `min_user_interactions` / `min_item_interactions` or load more rows. | 5/10 thresholds can be aggressive on subsamples. |
| **Wide category/author features** | Prefer learned embeddings (e.g., 32–64 dims) over one‑hot. | Reduces params; captures semantic similarity. |
| **Using SparseAdam** | Ensure the embedding layers are created with `sparse=True`. | Wrong combo will break training. |
| **Sharing reports** | Convert Markdown to HTML/PDF; automate in CI. | Improves visibility across the team. |

---

## 7) Quick Metrics Dashboard

| Metric | Meaning | Tracking suggestion |
|---|---|---|
| **Recall@K / Precision@K** | Top‑K correctness. | Compare against popularity baselines. |
| **NDCG@K** | How high true items rank. | If dropping, revisit scoring function/negatives. |
| **Hit Rate@K** | At least one correct item in Top‑K. | Proxy for user satisfaction. |
| **MAP@K** | Overall ranking quality. | Especially useful for long‑tail users. |
| **Feature Corr (r/p)** | Relation between metadata and scores. | Drop features with weak or noisy signals (high p). |
| **Category/Author Match** | Alignment with user history. | Tune adaptive‑mimic and metadata pipeline. |

---

## 8) Next Enhancements

1. Switch category/author to embedding features with hashing/regularization strategies.  
2. Automate checkpoints, early stopping, and hyper‑parameter sweeps.  
3. Experiment tracking (MLflow/W&B) + push reports into CI pipelines.  
4. Serving: export towers + index maps behind a REST/gRPC API.

---

## 9) Glossary

- **Two‑Tower Model:** Recommender architecture that compares user/item embeddings via similarity.  
- **Negative Sampling:** Generate random mismatches to teach the model “this interaction didn’t happen.”  
- **Adaptive Mimic:** A gating mechanism that blends metadata and ID embeddings dynamically.  
- **Hybrid Optimization:** Train dense parameters (Adam/AdamW) and sparse embeddings (SparseAdam) together.  
- **Ranking Metrics:** Metrics for recommendation quality (Recall, Precision, NDCG, MAP, Hit Rate).  
- **Feature Correlation Test:** Pearson r/p‑value between scores and features to detect drivers and side‑effects.

---

## 10) Quick Start Checklist

1. Install dependencies (CPU default):
   ```bash
   pip install -e .[dev]

   # GPU-enabled (use the CUDA tag that matches your driver)
   pip install -e .[dev,gpu] --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   Use the PyTorch install matrix to pick the correct CUDA wheel index (cu118, cu121, cu124, ...).
2. Put `books.csv` & `users.csv` into `data/`.  
3. Preprocess: `python scripts/preprocess.py --config configs/default.yaml` (verify user/item counts in logs).  
4. Train: `python scripts/train.py --config configs/default.yaml`  
5. Review the report at `artifacts/reports/recommendation_report.md` — metrics, embedding diagnostics, feature‑correlation tables, and sample recommendations.

You’re ready—happy experimenting!
