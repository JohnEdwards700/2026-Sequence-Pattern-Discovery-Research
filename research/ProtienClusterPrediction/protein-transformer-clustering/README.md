# Protein Transformer Clustering

This project implements a transformer model to cluster protein sequences based on resistance patterns. The pipeline includes data loading, tokenization, model training, and evaluation of clustering results.

## Project Structure

```
protein-transformer-clustering
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── fasta_loader.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── model
│   │   ├── transformer.py
│   │   ├── embedding.py
│   │   └── projection.py
│   ├── training
│   │   ├── train.py
│   │   ├── dataloader.py
│   │   ├── losses.py
│   │   └── scheduler.py
│   ├── evaluation
│   │   ├── cluster_analysis.py
│   │   └── metrics.py
│   └── utils
│       ├── io.py
│       └── config.py
├── configs
│   └── default.yaml
├── data
│   ├── raw
│   └── processed
├── notebooks
│   ├── 01-exploration.ipynb
│   └── 02-training.ipynb
├── scripts
│   ├── run_training.sh
│   └── eval_clusters.sh
├── models
│   └── README.md
├── tests
│   ├── test_tokenizer.py
│   └── test_model.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd protein-transformer-clustering
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Place your raw protein sequence data in the `data/raw` directory. The data should be in FASTA format.

2. **Tokenization**: The tokenizer will convert protein sequences into tokenized representations suitable for the transformer model.

3. **Training**: Run the training script to train the transformer model on the protein sequences. You can adjust hyperparameters in the `configs/default.yaml` file.

4. **Evaluation**: After training, use the evaluation scripts to analyze clustering results and visualize the patterns.

## Notebooks

The project includes Jupyter notebooks for exploration and training:

- `01-exploration.ipynb`: Exploratory data analysis and visualization of protein sequences.
- `02-training.ipynb`: Training process for the transformer model, including hyperparameter tuning.

## Scripts

- `run_training.sh`: Automates the training process.
- `eval_clusters.sh`: Automates the evaluation of clustering results.

## Testing

Unit tests are provided in the `tests` directory to ensure the functionality of the tokenizer and model implementations.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.