# Two-Tower Augmented with Adaptive Mimic Mechanism

Lean project scaffold for building a production-ready two-tower recommender tailored to the Amazon books dataset (`books.csv`, `users.csv`) in `data/`.

## Project Goals
- Offer a modular, testable codebase that separates data preparation, feature engineering, model definition, and training pipelines.
- Provide configuration-driven workflows for experimentation at scale.
- Keep the implementation approachable while leaving room for domain-specific extensions.
- Leverage book/user metadata (numeric statistics, categories, lightweight text cues) alongside ID embeddings for richer tower representations.

## Repository Layout
- `src/`: Library code organised by responsibility (`data`, `models`, `pipelines`, `utils`).
- `scripts/`: Command-line entry points for preprocessing and training routines.
- `configs/`: Experiment and environment configuration files.
- `tests/`: Lightweight sanity checks to safeguard core functionality.
- `docs/`: High-level design notes and ADRs.
- `data/`: Source CSVs (`books.csv`, `users.csv`) kept read-only, plus `*_trimmed.csv` samples for quick inspection.

## Getting Started
1. Create and activate a Python environment (>=3.10 recommended).
2. Install dependencies via `pip install -e .[dev]` to pull the CPU-only toolchain (PyTorch CPU build plus lint/test tooling). Add the `gpu` extra and point pip at the CUDA wheel index if you need GPU acceleration.
3. Run `python scripts/preprocess.py --config configs/default.yaml` to inspect indexed artefacts (serialization hooks forthcoming).
4. Train and evaluate the model using `python scripts/train.py --config configs/default.yaml`.

## Quick Setup Guide
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd two-tower-augmented-with-adaptive-mimic-mechanism
   ```
2. **Create & activate a virtualenv**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   # CPU-only (default PyPI wheels)
   pip install -e .[dev]

   # GPU-enabled (pick the CUDA version that matches your driver)
   pip install -e .[dev,gpu] --extra-index-url https://download.pytorch.org/whl/cu121
   ```
   > The `gpu` extra expects the PyTorch-provided CUDA wheels. Adjust the `--extra-index-url` to the appropriate CUDA tag (cu118, cu121, cu124, ...) per the [official install matrix](https://pytorch.org/get-started/locally/).
   > Need only runtime dependencies? Use `pip install -e .[cpu]` instead of the `dev` extra.
4. **Place data files**  
   Copy `books.csv` and `users.csv` (or the trimmed samples) into the `data/` directory—these large files are not tracked in git.
5. **Preprocess & sanity-check**
   ```bash
   python scripts/preprocess.py --config configs/default.yaml
   ```
   Inspect the logs to confirm user/item/interaction counts and feature dimensions.
6. **Train & evaluate**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```
   Results (ranking metrics, embedding diagnostics, recommendation samples) are reported in `artifacts/reports/recommendation_report.md`.

### Data Sources
- Full-scale training uses `data/books.csv` and `data/users.csv` by default.
- Set `data.books_limit` / `data.interactions_limit` to cap ingested rows (default: 2M books, 200K interactions) for quicker experiments.
- For quick smoke tests, duplicate `configs/default.yaml`, set `data.books_file` / `data.users_file` to the trimmed variants, and point the CLI `--config` flag at the new file.
- Metadata feature engineering is controlled via `data.feature_params` (numeric columns, category/author caps, user aggregation strategy).
- Default tower embeddings use 96 dimensions (with 192→96 feature MLPs); tune `model.*` blocks in `configs/default.yaml` to experiment with larger or smaller representations.
- Use `data.min_user_interactions` / `data.min_item_interactions` (defaults: 5 & 10) to prune cold users/items before training.

### Evaluation Outputs
- Offline ranking metrics (recall/precision/NDCG/MAP) are computed via candidate sampling and logged after training.
- Embedding diagnostics (norm summaries, neighbor category overlap, user-feature alignment) and qualitative recommendation audits are written to `artifacts/reports/recommendation_report.md`.
- Recommendation previews include category/author match rates against each sampled user's historical interests.

## Next Steps
- Benchmark large-scale runs (SparseAdam vs. hybrid optimisers) and tune embedding/feature dimensions.
- Add checkpointing, early stopping, and hyperparameter sweeps to `src/pipelines/training.py`.
- Expand automated tests and add fixtures covering recommendation reports, per-user metrics, and embedding diagnostics.

## A Critique of The Model
https://chatgpt.com/s/68fa36e8f18c8191a04447aca5e9e98d
