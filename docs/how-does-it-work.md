# How Does It Work?

This document walks through the full life-cycle of the project—from raw CSV files to evaluation reports—explaining both the technical mechanics and the broader reasoning behind each component. Use it as a guided tour when onboarding teammates, planning experiments, or debugging the pipeline.

---

## 1. Big‑Picture Flow

1. **Dataset Ingestion**  
   Raw Amazon-style CSVs (`books.csv`, `users.csv`) are loaded. Optional limits and filters reduce volume for faster experiments.
2. **Preprocessing & Feature Engineering**  
   Users/items are indexed, low-activity records are filtered, and multiple feature families (numeric, text stats, categorical signals) are transformed into tensors.
3. **Model Construction**  
   Modern two-tower architecture with configurable towers (`src/models/encoders.py`). Towers blend ID embeddings with metadata encoders via an adaptive mimic fusion module.
4. **Training Loop**  
   Negative sampling feeds a sampled-softmax style BCE objective. Hybrid optimisers (dense + sparse) keep giant embedding tables manageable.
5. **Evaluation & Reporting**  
   Ranking metrics, embedding diagnostics, and qualitative recommendation audits are computed and saved to `artifacts/reports/recommendation_report.md`.

---

## 2. Data Layer

| Component | Purpose | Key Details |
|-----------|---------|-------------|
| `data/books.csv`, `data/users.csv` | Raw inputs | Potentially multi-gigabyte files. Trimmed variants (`*_trimmed.csv`) exist for quick tests. |
| `configs/default.yaml:data` | Configuration | Set file roots, row limits, min interaction thresholds, and feature options. |
| `src/data/loaders.py` | CSV readers | Accept optional `books_limit`/`interactions_limit`. Filter out interactions referencing books outside the loaded subset. |
| `src/data/preprocessing.py` | Core transformations | 1. Drop invalid rows.<br>2. Iteratively remove users with &lt; `min_user_interactions` and items with &lt; `min_item_interactions`.<br>3. Build consistent index mappings + derived feature matrices.<br>4. Return a `TrainingDataset` dataclass. |
| `src/data/features.py` | Feature engineering | Produce numeric z-scores, lightweight text statistics, and categorical signals (currently one-hot). Also aggregate per-user features by pooling item vectors. |
| `src/data/indexers.py` / `datasets.py` / `samplers.py` | Utilities | Index maps, `torch.utils.data.Dataset` wrapper, and negative sampling logic. |

### Non-Technical Summary
A combination of rules and transformations tidies the raw data: we throw away rare users/books (to reduce noise), normalise metadata, and convert everything into array form so PyTorch can train efficiently.

---

## 3. Model Layer

### Two-Tower Architecture (`src/models`)

1. **ID Embeddings** (`build_id_embedding`)  
   Large embedding tables (now 96 dimensions by default) store latent representations for each user and item ID. Declaring `sparse: true` keeps optimizer memory in check.

2. **Feature Encoders** (`build_feature_encoder`)  
   Metadata vectors (numeric stats, text-derived features, categorical slots) flow through a configurable MLP (default: 192 → 96 with dropout). Supports alternative encoders (linear, identity, etc.).

3. **Adaptive Mimic Fusion** (`AdaptiveMimicModule`)  
   A small gating network blends ID embeddings with feature encoder outputs, letting metadata influence cold-start situations without overpowering dense collaborative signals.

4. **Tower Wrapper** (`TowerEncoder`)  
   Combines the above pieces. Fusion modes include `identity`, `sum`, `concat`, or `adaptive_mimic`. The tower exposes a consistent forward API and tracks its output dimension for downstream logging.

5. **Two-Tower Forward** (`TwoTowerModel`)  
   Receives user and item inputs (indices plus optional metadata tensors), returns embeddings, and applies the configured similarity metric (cosine or dot product).

### Why This Design?
- **Modularity:** Swap encoders/fusion strategies from config without rewriting training logic.
- **Cold-start robustness:** Adaptive mimic ensures the model isn’t helpless when faced with new or sparse entities.
- **Scalability:** Sparse embeddings + dense feature MLPs balance capacity and memory.

---

## 4. Training & Evaluation Pipeline

### Training Steps (`src/pipelines/training.py`)

1. **Seed Control**  
   `experiment.seed` ensures reproducible dataloading/sampling when desired.

2. **Dataset Assembly**  
   Calls into the data layer with feature and frequency thresholds. Logs counts and feature dimensions for quick sanity checks.

3. **Train/Validation Split**  
   Latest interaction per user becomes validation by default (`evaluation.holdout`). If timestamps are missing, training proceeds without evaluation.

4. **Tensor Preparation**  
   Feature matrices are moved to the chosen device (`cpu`/`cuda`). Towers are constructed via config -> builder pathway.

5. **Optimisation Setup**  
   - Dense parameters (MLPs, mimic gates) → `torch.optim.Adam` (or alternatives).  
   - Sparse parameters (embedding weights) → `torch.optim.SparseAdam`.  
   Hybrid lists keep zeroing/backprop logic simple.

6. **Mini-batch Loop**  
   - Sample negatives on the fly (`src/data/samplers.py`).  
   - Compute user/item embeddings, positive/negative logits.  
   - Use `BCEWithLogitsLoss` with concatenated labels (ones for positives, zeros for negatives).  
   - Backprop and step each optimizer.

7. **Evaluation**  
   After training, `_evaluate_model` generates candidate sets (mix of validation positives + random negatives) and scores them. The aggregated metrics (recall/precision/NDCG/MAP) get logged.

8. **Diagnostics & Reporting**  
   - Embedding norms, neighbor overlap, user-feature alignment.  
   - Recommendation audits: sample users, log top-K items, and compare category/author matches with historical tastes.  
   - Markdown report saved to `artifacts/reports/recommendation_report.md`.

### Non-Technical Summary
Think of it as teaching two “neurons” (user tower and item tower) to speak the same language. We repeatedly show the model examples of actual purchases (positives) vs. random pairings (negatives), and it learns to push related user/item vectors closer together. Afterwards, we check the quality with ranking scores and human-readable recommendation summaries.

---

## 5. Configuration Reference (key fields)

```yaml
data:
  books_file: "books.csv"
  users_file: "users.csv"
  books_limit: null                # None = load entire file; set int to cap rows
  interactions_limit: null
  min_user_interactions: 5         # Drop users with < 5 interactions
  min_item_interactions: 10        # Drop items with < 10 interactions
  feature_params:
    numeric_columns: [...]
    category_top_k: 300
    author_top_k: 300
    user_aggregation: "mean"

model:
  user_encoder:
    id_embedding: { params: { embedding_dim: 96, sparse: true }, ... }
    feature_encoder: { type: "mlp", hidden_dims: [192], output_dim: 96, dropout: 0.1 }
    fusion: "adaptive_mimic"
  item_encoder: { ... }            # mirrors user config
  similarity: "cosine"
  device: "cpu"

training:
  batch_size: 512
  num_epochs: 10
  optimizer: "adam"
  negatives_per_positive: 5

evaluation:
  metrics_k: [5, 10, 20]
  candidate_samples: 500

recommendations:
  sample_users: 3
  top_k: 5

diagnostics:
  item_sample_size: 500
  user_sample_size: 5000
  neighbor_k: 10
  report_path: "artifacts/reports/recommendation_report.md"
```

Adjusting these values is the primary way to explore speed/quality trade-offs.

---

## 6. Practical Tips & Caveats

- **Data Volume**  
  Huge CSVs will stress RAM; the row limits (`books_limit`, `interactions_limit`) are there for a reason. When you remove them, ensure the machine has capacity.

- **Frequency Thresholds**  
  On heavily subsampled data, `min_user_interactions=5` | `min_item_interactions=10` might prune everything. Lower them or ingest more rows.

- **Feature Width**  
  The current 300 category/author one-hot slots are a placeholder. Consider replacing them with learned embeddings for better generalisation and lower dimensionality.

- **Sparse Optimizer**  
  `SparseAdam` needs sparse gradients (`embedding(..., sparse=True)`). If you switch to dense embeddings, revisit optimizer settings.

- **Evaluation Sampling**  
  Candidate sampling speeds up evaluation but introduces variance. Increase `candidate_samples` for tighter estimates at the cost of compute.

- **Reports**  
  The Markdown report is intentionally simple—feel free to convert it to HTML or plug into dashboards.

---

## 7. Next Enhancements

1. **Embedding Category/Author Features**  
   Move away from one-hot vectors; store categorical IDs and learn compact embeddings.
2. **Checkpointing & Early Stopping**  
   Save intermediate weights, monitor validation, and stop early when metrics stagnate.
3. **Experiment Tracking**  
   Integrate MLflow/W&B for parameter sweeps and richer analytics.
4. **Serving Path**  
   Package trained towers + index mapping for real-time inference.

---

## 8. Glossary

- **Two-Tower Model:** Architecture with separate user and item encoders whose outputs are compared via a similarity function.
- **Negative Sampling:** Technique to create artificial “non-interactions” for contrastive learning.
- **Adaptive Mimic:** Gated fusion that lets metadata guide embeddings, especially useful for cold-start.
- **Hybrid Optimisation:** Using both dense (Adam/AdamW) and sparse (SparseAdam) optimisers simultaneously.
- **Ranking Metrics:** Measures (recall, precision, NDCG, MAP, hit-rate) that judge recommendation quality.

---

## 9. Quick Start Checklist

1. Install dependencies: `pip install -e .[dev]`.
2. Verify dataset access: `python scripts/preprocess.py --config configs/default.yaml`.
3. Train: `python scripts/train.py --config configs/default.yaml`.
4. Inspect logs; read `artifacts/reports/recommendation_report.md`.
5. Tweak config (limits, thresholds, feature params) and iterate.

With this overview, you should understand both the “why” and the “how” of the system. Happy experimenting!

