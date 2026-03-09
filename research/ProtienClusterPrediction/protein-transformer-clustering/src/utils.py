"""
Utilities for Protein Transformer Clustering.

This module consolidates common utility functions:
- Configuration: Loading and managing YAML configs
- File I/O: Saving/loading models and results
- General Helpers: Common operations used across modules

Connections to Other Modules:
    - Used by: model.py, data.py, training.py, evaluation.py
    - Central utility module with no internal project dependencies
"""

import os
import yaml
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import Counter

# ============================================================
# SECTION: CONFIGURATION MANAGEMENT
# Purpose: Load and manage YAML configuration files
# Dependencies: pyyaml
# Used by: training.py, evaluation.py, model.py
# ============================================================

class Config:
    """
    Configuration manager for loading and accessing YAML configs.
    
    Role in Pipeline:
        Central configuration handling - loads hyperparameters and settings
        from YAML files and provides convenient access methods.
    
    Connections:
        - Input from: YAML config files (configs/*.yaml)
        - Output to: All modules needing configuration
    
    Args:
        config_file: Path to YAML configuration file
    
    Example:
        >>> config = Config('configs/default.yaml')
        >>> batch_size = config['training']['batch_size']
        >>> embed_dim = config.get('embedding_dim', 128)
    """
    def __init__(self, config_file: str = 'configs/default.yaml'):
        self.config_file = config_file
        self.params = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary of configuration parameters
        
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_file}"
            )
        
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key: Configuration key (supports nested keys with '.')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get('training.batch_size', 32)
        """
        keys = key.split('.')
        value = self.params
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def __getitem__(self, key: str) -> Any:
        """
        Dictionary-style access to configuration.
        
        Args:
            key: Top-level configuration key
        
        Returns:
            Configuration value
        """
        return self.params.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self.params
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.params.copy()
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to a new YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load config without creating Config object.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(
    base_config: Dict[str, Any], 
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Role in Pipeline:
        Allows combining default config with experiment-specific overrides.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override values (takes precedence)
    
    Returns:
        Merged configuration dictionary
    
    Example:
        >>> default = {'model': {'dim': 128}, 'training': {'lr': 0.001}}
        >>> override = {'training': {'lr': 0.01}}
        >>> merged = merge_configs(default, override)
        >>> # merged['training']['lr'] == 0.01
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


# ============================================================
# SECTION: MODEL I/O
# Purpose: Save and load model checkpoints
# Dependencies: torch
# Used by: training.py, evaluation.py
# ============================================================

def save_model(
    model: torch.nn.Module, 
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint with optional training state.
    
    Role in Pipeline:
        Checkpoint saving during/after training for later evaluation
        or resuming training.
    
    Connections:
        - Input from: training.py (trained model)
        - Output to: File system (checkpoint files)
    
    Args:
        model: PyTorch model to save
        filepath: Path to save checkpoint
        optimizer: Optional optimizer state to include
        epoch: Optional current epoch number
        config: Optional configuration to include
    
    Example:
        >>> save_model(model, 'checkpoints/best.pt', optimizer, epoch=50)
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    model: torch.nn.Module, 
    filepath: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint and optionally restore optimizer state.
    
    Role in Pipeline:
        Load trained models for evaluation or continue training.
    
    Connections:
        - Input from: Saved checkpoint files
        - Output to: evaluation.py, training.py
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optional optimizer to restore state to
        device: Device to map tensors to
    
    Returns:
        Dictionary with any extra checkpoint data (epoch, config, etc.)
    
    Example:
        >>> info = load_model(model, 'checkpoints/best.pt')
        >>> print(f"Loaded from epoch {info.get('epoch')}")
    """
    device = device or torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    
    # Return any extra info
    extra = {k: v for k, v in checkpoint.items() 
             if k not in ['model_state_dict', 'optimizer_state_dict']}
    return extra


def save_model_simple(model: torch.nn.Module, filepath: str) -> None:
    """
    Save only model state dict (lightweight checkpoint).
    
    Args:
        model: PyTorch model
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def load_model_simple(model: torch.nn.Module, filepath: str) -> None:
    """
    Load only model state dict.
    
    Args:
        model: PyTorch model to load into
        filepath: Path to state dict
    """
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()


# ============================================================
# SECTION: RESULTS I/O
# Purpose: Save and load clustering results and metrics
# Dependencies: json, standard I/O
# Used by: evaluation.py
# ============================================================

def save_cluster_results(
    cluster_results: Dict[int, Dict], 
    filepath: str
) -> None:
    """
    Save cluster analysis results to file.
    
    Role in Pipeline:
        Persist cluster summaries for later analysis or reporting.
    
    Connections:
        - Input from: evaluation.py (summarize_clusters output)
        - Output to: Text file for human reading
    
    Args:
        cluster_results: Dictionary from summarize_clusters()
        filepath: Output file path
    
    Example:
        >>> save_cluster_results(summary, 'results/clusters.txt')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for cluster_id, data in sorted(cluster_results.items()):
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Count: {data.get('count', 'N/A')}\n")
            
            if 'mean_length' in data:
                f.write(f"  Mean Length: {data['mean_length']:.1f}\n")
            
            if 'gc_mean' in data:
                f.write(f"  GC Content: {data['gc_mean']:.3f}\n")
            
            f.write("\n")
    
    print(f"Cluster results saved to {filepath}")


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metric names to values
        filepath: Output JSON path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    clean_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            clean_metrics[key] = value.item()
        elif value is None:
            clean_metrics[key] = None
        else:
            clean_metrics[key] = float(value) if isinstance(value, (int, float)) else value
    
    with open(filepath, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to JSON metrics file
    
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================
# SECTION: GENERAL UTILITIES
# Purpose: Common helper functions
# Dependencies: None
# Used by: All modules
# ============================================================

def setup_device(use_cuda: bool = True) -> torch.device:
    """
    Setup and return appropriate compute device.
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        torch.device for computation
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n: int) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        n: Number to format
    
    Returns:
        Formatted string (e.g., "1.5M")
    """
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# SECTION: LOGGING UTILITIES
# Purpose: Training/evaluation logging helpers
# Dependencies: None
# Used by: training.py, evaluation.py
# ============================================================

class TrainingLogger:
    """
    Simple training logger for tracking metrics over time.
    
    Role in Pipeline:
        Records training/validation metrics during training for
        later analysis and visualization.
    
    Args:
        log_dir: Directory to save logs
    
    Example:
        >>> logger = TrainingLogger('logs/')
        >>> logger.log('train_loss', 0.5, epoch=1)
        >>> logger.save()
    """
    def __init__(self, log_dir: Optional[str] = None):
        self.history = {}
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step/epoch number
        """
        if name not in self.history:
            self.history[name] = []
        
        entry = {'value': value}
        if step is not None:
            entry['step'] = step
        
        self.history[name].append(entry)
    
    def get_history(self, name: str) -> list:
        """Get history for a metric."""
        return self.history.get(name, [])
    
    def save(self, filename: str = 'training_log.json') -> None:
        """Save log to JSON file."""
        if self.log_dir is None:
            print("No log directory specified")
            return
        
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training log saved to {filepath}")
    
    def load(self, filename: str = 'training_log.json') -> None:
        """Load log from JSON file."""
        if self.log_dir is None:
            return
        
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.history = json.load(f)
