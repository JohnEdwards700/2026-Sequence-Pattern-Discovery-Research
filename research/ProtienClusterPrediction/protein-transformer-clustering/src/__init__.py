"""
Protein Transformer Clustering - Simplified Module Structure.

This package provides a complete pipeline for clustering protein sequences
using transformer-based embeddings.

Modules:
    - model: Neural network components (Transformer, Embedding, Projection)
    - data: Data loading and preprocessing (FASTA, Tokenizer, Dataset)
    - training: Training loop, losses, and schedulers
    - evaluation: Clustering metrics and analysis
    - utils: Configuration, I/O, and helper functions

Quick Start:
    >>> from src.model import Transformer, create_model
    >>> from src.data import ProteinTokenizer, ProteinDataset, load_fasta_sequences
    >>> from src.training import Trainer, ContrastiveLoss
    >>> from src.evaluation import Evaluator, calculate_all_metrics
    >>> from src.utils import Config, save_model, load_model

Pipeline Flow:
    1. Load data: FASTA → Tokenizer → Dataset → DataLoader
    2. Create model: Config → Transformer
    3. Train: Trainer(model, data, loss) → trained model
    4. Evaluate: Embeddings → Clustering → Metrics

For detailed usage, see README.md
"""

# Version info
__version__ = "1.0.0"
__author__ = "Protein Transformer Clustering Team"

# ============================================================
# Core Model Components
# ============================================================
from .model import (
    PositionalEncoding,
    ProteinEmbedding,
    ProjectionLayer,
    Transformer,
    create_model,
)

# ============================================================
# Data Pipeline Components
# ============================================================
from .data import (
    ProteinTokenizer,
    ProteinDataset,
    UnlabeledProteinDataset,
    load_fasta_sequences,
    load_fasta_sequences_only,
    load_all_fasta_from_directory,
    create_dataset_from_fasta,
    collate_fn,
)

# ============================================================
# Training Components
# ============================================================
from .training import (
    ContrastiveLoss,
    ClusteringLoss,
    CustomLoss,
    WarmupScheduler,
    get_scheduler,
    ProteinDataLoader,
    Trainer,
    train_model,
)

# ============================================================
# Evaluation Components
# ============================================================
from .evaluation import (
    calculate_silhouette_score,
    calculate_adjusted_rand_index,
    calculate_normalized_mutual_info,
    calculate_davies_bouldin_score,
    calculate_all_metrics,
    get_cluster_distribution,
    plot_cluster_distribution,
    summarize_clusters,
    print_cluster_summary,
    extract_embeddings,
    run_clustering,
    Evaluator,
    evaluate_model,
)

# ============================================================
# Utility Components
# ============================================================
from .utils import (
    Config,
    load_config,
    merge_configs,
    save_model,
    load_model,
    save_model_simple,
    load_model_simple,
    save_cluster_results,
    save_metrics,
    load_metrics,
    setup_device,
    set_seed,
    count_parameters,
    format_number,
    ensure_dir,
    TrainingLogger,
)

# ============================================================
# Public API
# ============================================================
__all__ = [
    # Model
    "PositionalEncoding",
    "ProteinEmbedding", 
    "ProjectionLayer",
    "Transformer",
    "create_model",
    # Data
    "ProteinTokenizer",
    "ProteinDataset",
    "UnlabeledProteinDataset",
    "load_fasta_sequences",
    "load_fasta_sequences_only",
    "load_all_fasta_from_directory",
    "create_dataset_from_fasta",
    "collate_fn",
    # Training
    "ContrastiveLoss",
    "ClusteringLoss",
    "CustomLoss",
    "WarmupScheduler",
    "get_scheduler",
    "ProteinDataLoader",
    "Trainer",
    "train_model",
    # Evaluation
    "calculate_silhouette_score",
    "calculate_adjusted_rand_index",
    "calculate_normalized_mutual_info",
    "calculate_davies_bouldin_score",
    "calculate_all_metrics",
    "get_cluster_distribution",
    "plot_cluster_distribution",
    "summarize_clusters",
    "print_cluster_summary",
    "extract_embeddings",
    "run_clustering",
    "Evaluator",
    "evaluate_model",
    # Utils
    "Config",
    "load_config",
    "merge_configs",
    "save_model",
    "load_model",
    "save_model_simple",
    "load_model_simple",
    "save_cluster_results",
    "save_metrics",
    "load_metrics",
    "setup_device",
    "set_seed",
    "count_parameters",
    "format_number",
    "ensure_dir",
    "TrainingLogger",
]