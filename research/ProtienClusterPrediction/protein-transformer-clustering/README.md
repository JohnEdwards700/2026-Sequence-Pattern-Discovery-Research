# ProtienClusterPrediction

Compact overview and quickstart for the protein-transformer-clustering project.

## Project purpose
- Learn sequence embeddings for protein sequences using a transformer backbone and a projection head, then cluster embeddings to discover protein families/relationships.

## Layout (key folders)
- `protein-transformer-clustering/`: main package with code, configs, data, models, notebooks, and scripts.
- `pipeline.py`: top-level orchestration script that wires preprocessing, training, and evaluation.

Inside `protein-transformer-clustering/`:
- `configs/default.yaml`: experiment and training configuration.
- `data/raw` and `data/processed`: raw FASTA inputs and processed/tokenized outputs.
- `src/`: core Python modules:
  - `src/data`: `fasta_loader.py`, `tokenizer.py`, `dataset.py` — parsing, tokenization, and dataset wrappers.
  - `src/model`: `transformer.py`, `embedding.py`, `projection.py` — model backbone and projection head used for clustering.
  - `src/training`: `train.py`, `dataloader.py`, `losses.py`, `scheduler.py` — training loop, losses, and utilities.
  - `src/evaluation`: `cluster_analysis.py`, `metrics.py` — cluster evaluation and metrics.
  - `src/utils`: `config.py`, `io.py` — config loader and IO helpers.
- `notebooks/`: EDA and interactive training notebooks.
- `scripts/`: convenience shell scripts like `run_training.sh` and `eval_clusters.sh`.
- `models/`: place to save trained checkpoints and model artifacts.
- `tests/`: unit tests for tokenizer and model components.

## Quick start
1. Install dependencies (use a venv):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r protein-transformer-clustering/requirements.txt
```

2. Edit experiment settings in `protein-transformer-clustering/configs/default.yaml`.

3. Prepare data: place FASTA files in `protein-transformer-clustering/data/raw` and run the preprocessing/tokenization pipeline (see `src/data`).

4. Run training (example using the script):

```bash
cd protein-transformer-clustering
./scripts/run_training.sh configs/default.yaml
```

Or run the training entry directly:

```bash
python -m src.training.train --config configs/default.yaml
```

5. Evaluate clusters using the provided evaluation utilities or `scripts/eval_clusters.sh`.

## Tests
- Run unit tests with `pytest` from the `protein-transformer-clustering` folder.

## Notes & next steps
- For quick experimentation use the notebooks in `notebooks/`.
- `src/model/projection.py` implements the projection head that maps encoder outputs to the embedding space used for clustering — inspect it when tuning downstream clustering quality.

If you want, I can also:
- annotate `src/model/projection.py` with inline explanations,
- run the unit tests,
- or create a short CONTRIBUTING or developer README.
