"""
Evaluation and Metrics for Protein Transformer Clustering.

This module consolidates all evaluation-related components:
- Clustering Metrics: Silhouette score, ARI, NMI
- Cluster Analysis: Distribution plotting, summarization
- Evaluation Pipeline: Full evaluation workflow

Evaluation Flow:
    Trained Model → Extract Embeddings → Clustering → Metrics → Report

Connections to Other Modules:
    - Uses: model.py (trained model for embedding extraction)
    - Uses: data.py (datasets for evaluation)
    - Uses: utils.py (saving results)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

# Optional imports for clustering and visualization
try:
    from sklearn.metrics import (
        silhouette_score, 
        adjusted_rand_score, 
        normalized_mutual_info_score,
        davies_bouldin_score
    )
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some metrics will be disabled.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting will be disabled.")


# ============================================================
# SECTION: CLUSTERING METRICS
# Purpose: Quantify clustering quality
# Dependencies: scikit-learn
# Used by: evaluate_clustering, Evaluator
# ============================================================

def calculate_silhouette_score(
    embeddings: np.ndarray, 
    cluster_labels: np.ndarray
) -> Optional[float]:
    """
    Calculate silhouette score for clustering quality.
    
    Role in Pipeline:
        Primary metric for evaluating how well-separated clusters are.
        Higher values (closer to 1) indicate better-defined clusters.
    
    Connections:
        - Input from: Model embeddings + clustering assignments
        - Output to: Evaluation report
    
    Interpretation:
        - Score near +1: Clusters are well-separated
        - Score near 0: Clusters overlap
        - Score near -1: Samples may be in wrong clusters
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        cluster_labels: Array of cluster assignments (n_samples,)
    
    Returns:
        Silhouette score or None if invalid (e.g., single cluster)
    
    Example:
        >>> score = calculate_silhouette_score(embeddings, labels)
        >>> print(f"Silhouette: {score:.4f}")
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    # Need at least 2 clusters
    n_clusters = len(set(cluster_labels))
    if n_clusters < 2:
        return None
    
    return silhouette_score(embeddings, cluster_labels)


def calculate_adjusted_rand_index(
    true_labels: np.ndarray, 
    predicted_labels: np.ndarray
) -> float:
    """
    Calculate Adjusted Rand Index between true and predicted labels.
    
    Role in Pipeline:
        Measures agreement between ground truth and predicted clustering,
        adjusted for chance. Used when true labels are available.
    
    Connections:
        - Input from: Ground truth labels + predicted cluster assignments
        - Output to: Evaluation report
    
    Interpretation:
        - ARI = 1: Perfect agreement
        - ARI = 0: Random clustering
        - ARI < 0: Worse than random
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        Adjusted Rand Index score
    """
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    return adjusted_rand_score(true_labels, predicted_labels)


def calculate_normalized_mutual_info(
    true_labels: np.ndarray, 
    predicted_labels: np.ndarray
) -> float:
    """
    Calculate Normalized Mutual Information between labelings.
    
    Role in Pipeline:
        Information-theoretic metric measuring shared information
        between true and predicted clustering. Normalized to [0, 1].
    
    Connections:
        - Input from: Ground truth + predicted labels
        - Output to: Evaluation report
    
    Interpretation:
        - NMI = 1: Perfect agreement
        - NMI = 0: No mutual information
    
    Args:
        true_labels: Ground truth cluster labels
        predicted_labels: Predicted cluster labels
    
    Returns:
        Normalized Mutual Information score
    """
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    return normalized_mutual_info_score(true_labels, predicted_labels)


def calculate_davies_bouldin_score(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray
) -> Optional[float]:
    """
    Calculate Davies-Bouldin Index for cluster separation.
    
    Role in Pipeline:
        Measures average similarity between clusters. Lower values
        indicate better clustering (more separated, compact clusters).
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        cluster_labels: Cluster assignments
    
    Returns:
        Davies-Bouldin Index or None if invalid
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    n_clusters = len(set(cluster_labels))
    if n_clusters < 2:
        return None
    
    return davies_bouldin_score(embeddings, cluster_labels)


def calculate_all_metrics(
    embeddings: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, Optional[float]]:
    """
    Calculate all available clustering metrics.
    
    Role in Pipeline:
        Convenience function to compute all metrics at once
        for comprehensive evaluation reporting.
    
    Args:
        embeddings: Sequence embeddings
        predicted_labels: Predicted cluster assignments
        true_labels: Optional ground truth labels
    
    Returns:
        Dictionary of metric names to scores
    """
    metrics = {
        'silhouette_score': calculate_silhouette_score(embeddings, predicted_labels),
        'davies_bouldin_score': calculate_davies_bouldin_score(embeddings, predicted_labels),
    }
    
    if true_labels is not None:
        metrics['adjusted_rand_index'] = calculate_adjusted_rand_index(
            true_labels, predicted_labels
        )
        metrics['normalized_mutual_info'] = calculate_normalized_mutual_info(
            true_labels, predicted_labels
        )
    
    return metrics


# ============================================================
# SECTION: CLUSTER ANALYSIS
# Purpose: Analyze and visualize cluster properties
# Dependencies: matplotlib (optional), numpy
# Used by: Evaluator, analysis scripts
# ============================================================

def get_cluster_distribution(cluster_ids: np.ndarray) -> Dict[int, int]:
    """
    Get the count of samples in each cluster.
    
    Args:
        cluster_ids: Array of cluster assignments
    
    Returns:
        Dictionary mapping cluster ID to count
    """
    return dict(Counter(cluster_ids))


def plot_cluster_distribution(
    cluster_ids: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Cluster Distribution"
) -> None:
    """
    Plot bar chart of cluster sizes.
    
    Role in Pipeline:
        Visual inspection of cluster balance. Helps identify
        imbalanced clustering or dominant clusters.
    
    Connections:
        - Input from: Clustering assignments
        - Output to: Display or saved figure
    
    Args:
        cluster_ids: Array of cluster assignments
        save_path: Optional path to save figure
        title: Plot title
    
    Example:
        >>> plot_cluster_distribution(labels, save_path="clusters.png")
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting.")
        return
    
    cluster_counts = Counter(cluster_ids)
    clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in clusters]

    plt.figure(figsize=(10, 6))
    plt.bar(clusters, counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Sequences')
    plt.title(title)
    plt.xticks(clusters)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def summarize_clusters(
    cluster_ids: np.ndarray,
    sequences: Optional[List[str]] = None,
    headers: Optional[List[str]] = None,
    additional_features: Optional[Dict[str, np.ndarray]] = None
) -> Dict[int, Dict]:
    """
    Generate summary statistics for each cluster.
    
    Role in Pipeline:
        Detailed cluster analysis including counts, sequence properties,
        and any additional features provided.
    
    Connections:
        - Input from: Cluster assignments + optional metadata
        - Output to: Report generation, logging
    
    Args:
        cluster_ids: Array of cluster assignments
        sequences: Optional list of sequences for length analysis
        headers: Optional sequence identifiers
        additional_features: Dict of feature arrays (e.g., GC content)
    
    Returns:
        Dictionary mapping cluster ID to summary dict
    
    Example:
        >>> summary = summarize_clusters(labels, sequences)
        >>> for cid, info in summary.items():
        ...     print(f"Cluster {cid}: {info['count']} sequences")
    """
    cluster_ids = np.array(cluster_ids)
    unique_clusters = np.unique(cluster_ids)
    
    summary = {}
    
    for cid in unique_clusters:
        mask = cluster_ids == cid
        indices = np.where(mask)[0]
        
        cluster_info = {
            'count': int(mask.sum()),
            'indices': indices.tolist(),
        }
        
        # Add sequence statistics if available
        if sequences is not None:
            cluster_seqs = [sequences[i] for i in indices]
            lengths = [len(s) for s in cluster_seqs]
            cluster_info['mean_length'] = np.mean(lengths)
            cluster_info['min_length'] = min(lengths)
            cluster_info['max_length'] = max(lengths)
        
        # Add headers if available
        if headers is not None:
            cluster_info['headers'] = [headers[i] for i in indices]
        
        # Add additional features if provided
        if additional_features is not None:
            for feature_name, feature_values in additional_features.items():
                cluster_values = feature_values[mask]
                cluster_info[f'{feature_name}_mean'] = float(np.mean(cluster_values))
                cluster_info[f'{feature_name}_std'] = float(np.std(cluster_values))
        
        summary[int(cid)] = cluster_info
    
    return summary


def print_cluster_summary(summary: Dict[int, Dict]) -> None:
    """
    Pretty-print cluster summary to console.
    
    Args:
        summary: Output from summarize_clusters()
    """
    print("\n" + "=" * 50)
    print("CLUSTER SUMMARY")
    print("=" * 50)
    
    for cid, info in sorted(summary.items()):
        print(f"\nCluster {cid}:")
        print(f"  Number of sequences: {info['count']}")
        
        if 'mean_length' in info:
            print(f"  Mean sequence length: {info['mean_length']:.1f}")
            print(f"  Length range: [{info['min_length']}, {info['max_length']}]")
        
        # Print any additional numeric features
        for key, value in info.items():
            if key.endswith('_mean'):
                feature = key[:-5]
                std_key = f'{feature}_std'
                if std_key in info:
                    print(f"  {feature}: {value:.3f} ± {info[std_key]:.3f}")
    
    print("\n" + "=" * 50)


# ============================================================
# SECTION: EMBEDDING EXTRACTION
# Purpose: Extract embeddings from trained model
# Dependencies: model.py, torch
# Used by: Evaluator, run_clustering
# ============================================================

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    pooling: str = 'mean'
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Extract embeddings from a trained model.
    
    Role in Pipeline:
        Converts sequences to dense embeddings using the trained model.
        These embeddings are then used for clustering.
    
    Connections:
        - Input from: Trained model (model.py) + DataLoader (data.py)
        - Output to: Clustering algorithm
    
    Args:
        model: Trained transformer model
        dataloader: DataLoader yielding batches
        device: Device for computation
        pooling: Pooling strategy ('mean', 'max', 'cls')
    
    Returns:
        Tuple of (embeddings array, optional headers list)
    
    Example:
        >>> embeddings, headers = extract_embeddings(model, loader)
        >>> print(f"Extracted {len(embeddings)} embeddings")
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    all_headers = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        
        # Get attention mask for padding
        attention_mask = batch.get('attention_mask')
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).to(device)
        
        # Forward pass
        outputs = model(input_ids, src_key_padding_mask=padding_mask)
        
        # Pool to get single embedding per sequence
        if pooling == 'mean':
            # Mean pooling over sequence length
            if attention_mask is not None:
                # Mask-aware mean
                mask = attention_mask.unsqueeze(-1).float().to(device)
                embeddings = (outputs * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                embeddings = outputs.mean(dim=1)
        elif pooling == 'max':
            embeddings = outputs.max(dim=1)[0]
        elif pooling == 'cls':
            embeddings = outputs[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        all_embeddings.append(embeddings.cpu().numpy())
        
        # Collect headers if available
        if 'header' in batch:
            all_headers.extend(batch['header'])
    
    embeddings = np.vstack(all_embeddings)
    headers = all_headers if all_headers else None
    
    return embeddings, headers


# ============================================================
# SECTION: CLUSTERING
# Purpose: Apply clustering to embeddings
# Dependencies: scikit-learn
# Used by: Evaluator, main evaluation pipeline
# ============================================================

def run_clustering(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    method: str = 'kmeans',
    **kwargs
) -> np.ndarray:
    """
    Apply clustering algorithm to embeddings.
    
    Role in Pipeline:
        Core clustering step - groups similar embeddings together.
        Uses embeddings from model to find natural protein clusters.
    
    Connections:
        - Input from: extract_embeddings()
        - Output to: Metrics calculation, visualization
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans')
        **kwargs: Additional arguments for clustering algorithm
    
    Returns:
        Array of cluster assignments
    
    Example:
        >>> labels = run_clustering(embeddings, n_clusters=10)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for clustering")
    
    if method == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=kwargs.get('random_state', 42),
            n_init=kwargs.get('n_init', 10)
        )
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return labels


# ============================================================
# SECTION: EVALUATOR CLASS
# Purpose: Complete evaluation pipeline
# Dependencies: All above components
# Used by: Main evaluation script
# ============================================================

class Evaluator:
    """
    Complete evaluation pipeline for protein clustering.
    
    Role in Pipeline:
        End-to-end evaluation manager that:
        1. Extracts embeddings from trained model
        2. Applies clustering
        3. Computes all metrics
        4. Generates summary and visualizations
    
    Connections:
        - Uses: model.py (trained model)
        - Uses: data.py (evaluation dataset)
        - Uses: All metric and analysis functions above
    
    Args:
        model: Trained transformer model
        n_clusters: Number of clusters for K-means
        device: Computation device
    
    Example:
        >>> evaluator = Evaluator(model, n_clusters=10)
        >>> results = evaluator.evaluate(dataloader)
        >>> evaluator.print_report()
    """
    def __init__(
        self,
        model: torch.nn.Module,
        n_clusters: int = 5,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.n_clusters = n_clusters
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Results storage
        self.embeddings = None
        self.cluster_labels = None
        self.true_labels = None
        self.metrics = None
        self.summary = None
    
    def evaluate(
        self,
        dataloader: DataLoader,
        true_labels: Optional[np.ndarray] = None,
        sequences: Optional[List[str]] = None
    ) -> Dict:
        """
        Run full evaluation pipeline.
        
        Args:
            dataloader: DataLoader for evaluation data
            true_labels: Optional ground truth for supervised metrics
            sequences: Optional sequences for summary stats
        
        Returns:
            Dictionary containing all results
        """
        print("Extracting embeddings...")
        self.embeddings, headers = extract_embeddings(
            self.model, dataloader, self.device
        )
        print(f"Extracted {len(self.embeddings)} embeddings")
        
        print(f"Running K-means clustering (k={self.n_clusters})...")
        self.cluster_labels = run_clustering(
            self.embeddings, 
            n_clusters=self.n_clusters
        )
        
        print("Computing metrics...")
        self.true_labels = true_labels
        self.metrics = calculate_all_metrics(
            self.embeddings,
            self.cluster_labels,
            self.true_labels
        )
        
        print("Generating cluster summary...")
        self.summary = summarize_clusters(
            self.cluster_labels,
            sequences=sequences,
            headers=headers
        )
        
        return {
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'metrics': self.metrics,
            'summary': self.summary
        }
    
    def print_report(self) -> None:
        """Print evaluation report to console."""
        if self.metrics is None:
            print("No evaluation results. Run evaluate() first.")
            return
        
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)
        
        for metric_name, value in self.metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: N/A")
        
        if self.summary:
            print_cluster_summary(self.summary)
    
    def plot_results(self, save_dir: Optional[str] = None) -> None:
        """Generate and optionally save visualization plots."""
        if self.cluster_labels is None:
            print("No results to plot. Run evaluate() first.")
            return
        
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/cluster_distribution.png"
        
        plot_cluster_distribution(
            self.cluster_labels,
            save_path=save_path,
            title=f"Cluster Distribution (k={self.n_clusters})"
        )


# ============================================================
# SECTION: MAIN EVALUATION FUNCTION
# Purpose: Entry point for evaluation script
# Dependencies: All above, utils.py
# Used by: CLI, notebooks
# ============================================================

def evaluate_model(
    model_path: str,
    data_path: str,
    config_path: str,
    n_clusters: int = 5,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Main evaluation function - runs full pipeline from saved model.
    
    Role in Pipeline:
        Complete evaluation entry point - loads model, data, runs
        clustering, computes metrics, and saves results.
    
    Args:
        model_path: Path to saved model weights
        data_path: Path to evaluation data
        config_path: Path to model configuration
        n_clusters: Number of clusters
        output_dir: Directory to save results
    
    Returns:
        Dictionary of all evaluation results
    
    Example:
        >>> results = evaluate_model(
        ...     "models/best.pt",
        ...     "data/test.fasta",
        ...     "configs/default.yaml"
        ... )
    """
    import yaml
    from pathlib import Path
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Import and create model
    from .model import create_model
    model = create_model(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create dataset
    from .data import create_dataset_from_fasta, ProteinTokenizer
    tokenizer = ProteinTokenizer()
    dataset = create_dataset_from_fasta(data_path, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Run evaluation
    evaluator = Evaluator(model, n_clusters=n_clusters)
    results = evaluator.evaluate(dataloader)
    
    # Print report
    evaluator.print_report()
    
    # Save results if output dir specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        evaluator.plot_results(save_dir=output_dir)
        
        # Save metrics
        import json
        metrics_path = Path(output_dir) / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"Saved metrics to {metrics_path}")
    
    return results
