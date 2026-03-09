"""
Training Pipeline for Protein Transformer Clustering.

This module consolidates all training-related components:
- Loss Functions: ContrastiveLoss, ClusteringLoss, CustomLoss
- Learning Rate Scheduler: Warmup and decay scheduling
- DataLoader Wrapper: Convenient batching utility
- Training Loop: Main training orchestration

Training Flow:
    Config → Model + DataLoader + Loss + Optimizer → Training Loop → Saved Model

Connections to Other Modules:
    - Uses: model.py (Transformer model), data.py (ProteinDataset)
    - Uses: utils.py (Config loading, model saving)
    - Output to: evaluation.py (trained model for clustering)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer, Adam
import numpy as np
from typing import Dict, Optional, Callable, List, Tuple
import yaml
from pathlib import Path

# ============================================================
# SECTION: LOSS FUNCTIONS
# Purpose: Define training objectives for the model
# Dependencies: None (PyTorch base)
# Used by: TrainingLoop, train_model
# ============================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning similarity-based embeddings.
    
    Role in Pipeline:
        Primary loss for learning representations where similar sequences
        are pulled together and dissimilar sequences are pushed apart.
        Essential for meaningful clustering.
    
    Connections:
        - Input from: Model embeddings (two sequence embeddings + label)
        - Output to: Backpropagation for model updates
    
    How it works:
        - For similar pairs (label=0): Minimize distance
        - For dissimilar pairs (label=1): Push apart up to margin
    
    Args:
        margin: Minimum distance for dissimilar pairs (default: 1.0)
    
    Example:
        >>> loss_fn = ContrastiveLoss(margin=2.0)
        >>> loss = loss_fn(embed1, embed2, labels)
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self, 
        output1: torch.Tensor, 
        output2: torch.Tensor, 
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            output1: First embedding of shape (batch_size, embed_dim)
            output2: Second embedding of shape (batch_size, embed_dim)
            label: Binary labels - 0 for similar, 1 for dissimilar
        
        Returns:
            Scalar loss value
        """
        # Compute pairwise distances
        distance = nn.functional.pairwise_distance(output1, output2)
        
        # Contrastive loss formula
        loss = (1 - label) * torch.pow(distance, 2) + \
               label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        return loss.mean()


class ClusteringLoss(nn.Module):
    """
    Clustering-aware loss using distance to cluster centers.
    
    Role in Pipeline:
        Auxiliary loss that encourages embeddings to be close to their
        assigned cluster centers. Used in conjunction with contrastive loss
        during later training stages.
    
    Connections:
        - Input from: Model embeddings + cluster centers + cluster assignments
        - Output to: Combined loss for backpropagation
    
    Args:
        None
    
    Example:
        >>> loss_fn = ClusteringLoss()
        >>> loss = loss_fn(embeddings, centers, labels)
    """
    def __init__(self):
        super(ClusteringLoss, self).__init__()

    def forward(
        self, 
        embeddings: torch.Tensor, 
        cluster_centers: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clustering loss based on distance to centers.
        
        Args:
            embeddings: Sequence embeddings (batch_size, embed_dim)
            cluster_centers: Cluster centroids (n_clusters, embed_dim)
            labels: Cluster assignments (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # Compute distances to all cluster centers
        distances = torch.cdist(embeddings, cluster_centers)
        
        # Cross entropy over distances (softmax over negative distances)
        loss = nn.functional.cross_entropy(-distances, labels)
        
        return loss


class CustomLoss(nn.Module):
    """
    Combined loss function with configurable weights.
    
    Role in Pipeline:
        Flexible loss combining contrastive and clustering objectives.
        Allows tuning the balance between representation learning
        and cluster-aware training.
    
    Connections:
        - Uses: ContrastiveLoss, ClusteringLoss
        - Input from: Model outputs with all required components
        - Output to: Training loop for optimization
    
    Args:
        contrastive_weight: Weight for contrastive loss term
        clustering_weight: Weight for clustering loss term
    
    Example:
        >>> loss_fn = CustomLoss(contrastive_weight=1.0, clustering_weight=0.5)
        >>> loss = loss_fn(out1, out2, labels, embeddings, centers)
    """
    def __init__(
        self, 
        contrastive_weight: float = 1.0, 
        clustering_weight: float = 1.0
    ):
        super(CustomLoss, self).__init__()
        self.contrastive_loss = ContrastiveLoss()
        self.clustering_loss = ClusteringLoss()
        self.contrastive_weight = contrastive_weight
        self.clustering_weight = clustering_weight

    def forward(
        self, 
        output1: torch.Tensor, 
        output2: torch.Tensor, 
        pair_labels: torch.Tensor,
        embeddings: torch.Tensor, 
        cluster_centers: torch.Tensor,
        cluster_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined weighted loss.
        
        Args:
            output1: First pair embeddings
            output2: Second pair embeddings
            pair_labels: Similarity labels for pairs
            embeddings: All embeddings for clustering loss
            cluster_centers: Current cluster centroids
            cluster_labels: Cluster assignments
        
        Returns:
            Weighted sum of loss components
        """
        loss_contrastive = self.contrastive_loss(output1, output2, pair_labels)
        loss_clustering = self.clustering_loss(embeddings, cluster_centers, cluster_labels)
        
        return (self.contrastive_weight * loss_contrastive + 
                self.clustering_weight * loss_clustering)


# ============================================================
# SECTION: LEARNING RATE SCHEDULER
# Purpose: Control learning rate during training
# Dependencies: PyTorch Optimizer
# Used by: TrainingLoop, train_model
# ============================================================

class WarmupScheduler:
    """
    Learning rate scheduler with warmup and decay phases.
    
    Role in Pipeline:
        Controls learning rate throughout training:
        1. Warmup phase: Gradually increase LR from initial to peak
        2. Decay phase: Gradually decrease LR to final value
        
        Helps stabilize early training and fine-tune later stages.
    
    Connections:
        - Input from: Optimizer (modifies its learning rate)
        - Used by: Training loop (called after each step/epoch)
    
    Args:
        optimizer: PyTorch optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        initial_lr: Starting learning rate (during warmup)
        final_lr: Peak learning rate (at end of warmup) / end of training
    
    Example:
        >>> scheduler = WarmupScheduler(optimizer, warmup_steps=1000, total_steps=10000)
        >>> for step in range(10000):
        ...     train_step()
        ...     scheduler.step()
    """
    def __init__(
        self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        total_steps: int, 
        initial_lr: float = 1e-7,
        final_lr: float = 1e-3
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.current_step = 0

    def step(self):
        """Advance scheduler by one step and update learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """
        Calculate current learning rate based on step.
        
        Returns:
            Current learning rate value
        """
        if self.current_step < self.warmup_steps:
            # Linear warmup: initial_lr → final_lr
            progress = self.current_step / self.warmup_steps
            return self.initial_lr + (self.final_lr - self.initial_lr) * progress
        else:
            # Linear decay: final_lr → 0
            decay_steps = self.total_steps - self.warmup_steps
            decay_progress = (self.current_step - self.warmup_steps) / decay_steps
            return self.final_lr * (1 - decay_progress)

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr


def get_scheduler(
    optimizer: Optimizer, 
    config: Dict
) -> WarmupScheduler:
    """
    Factory function to create scheduler from config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration dict
    
    Returns:
        Configured WarmupScheduler instance
    """
    return WarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.get('warmup_steps', 1000),
        total_steps=config.get('total_steps', 10000),
        initial_lr=config.get('initial_lr', 1e-7),
        final_lr=config.get('learning_rate', 1e-3)
    )


# ============================================================
# SECTION: DATA LOADER WRAPPER
# Purpose: Convenient DataLoader creation for training
# Dependencies: PyTorch DataLoader, ProteinDataset
# Used by: train_model
# ============================================================

class ProteinDataLoader:
    """
    Wrapper for PyTorch DataLoader with protein-specific defaults.
    
    Role in Pipeline:
        Convenience class that wraps dataset in a DataLoader with
        sensible defaults for protein sequence training.
    
    Connections:
        - Input from: ProteinDataset (from data.py)
        - Output to: Training loop batches
    
    Args:
        dataset: PyTorch Dataset to wrap
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of parallel data loading workers
    
    Example:
        >>> loader = ProteinDataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     process(batch)
    """
    def __init__(
        self, 
        dataset: Dataset,
        batch_size: int = 32, 
        shuffle: bool = True,
        num_workers: int = 0
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers
        )

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)
    
    def get_loader(self) -> DataLoader:
        """Get the underlying DataLoader."""
        return self.dataloader


# ============================================================
# SECTION: TRAINING LOOP
# Purpose: Main training orchestration
# Dependencies: Model, DataLoader, Loss, Scheduler
# Used by: Main training script
# ============================================================

class Trainer:
    """
    Training orchestrator for protein transformer models.
    
    Role in Pipeline:
        Central training manager that coordinates:
        - Model forward/backward passes
        - Loss computation
        - Optimizer updates
        - Learning rate scheduling
        - Logging and checkpointing
    
    Connections:
        - Uses: model.py (Transformer), data.py (Dataset/DataLoader)
        - Uses: Loss functions and schedulers from this module
        - Output to: Saved model weights for evaluation.py
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        optimizer: PyTorch optimizer
        criterion: Loss function
        scheduler: Optional learning rate scheduler
        device: Device to train on (cuda/cpu)
    
    Example:
        >>> trainer = Trainer(model, train_loader, optimizer=optimizer, criterion=loss)
        >>> trainer.train(epochs=50)
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[WarmupScheduler] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Default optimizer if not provided
        self.optimizer = optimizer or Adam(model.parameters(), lr=1e-3)
        
        # Default loss if not provided
        self.criterion = criterion or ContrastiveLoss()
        
        self.scheduler = scheduler
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('label')
            if labels is not None:
                labels = labels.to(self.device)
            
            # Get attention mask if available
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
                # Convert to key padding mask (True = ignore)
                padding_mask = attention_mask == 0
            else:
                padding_mask = None
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, src_key_padding_mask=padding_mask)
            
            # Compute loss (simplified - assumes outputs and labels are compatible)
            if labels is not None:
                loss = self.criterion(outputs, labels)
            else:
                # For unsupervised, use reconstruction or other loss
                loss = outputs.mean()  # Placeholder
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update scheduler if using step-level scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('label')
            if labels is not None:
                labels = labels.to(self.device)
            
            attention_mask = batch.get('attention_mask')
            padding_mask = None
            if attention_mask is not None:
                padding_mask = (attention_mask == 0).to(self.device)
            
            outputs = self.model(input_ids, src_key_padding_mask=padding_mask)
            
            if labels is not None:
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
            
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self, 
        epochs: int,
        save_path: Optional[str] = None,
        log_interval: int = 1
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs to train
            save_path: Path to save best model checkpoint
            log_interval: Print progress every N epochs
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} "
                      f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
        
        print("Training complete.")
        return self.history


# ============================================================
# SECTION: TRAINING ENTRY POINT
# Purpose: Main function to run training from config
# Dependencies: All above components, utils.py, data.py, model.py
# Used by: Main script, CLI
# ============================================================

def train_model(config_path: str) -> nn.Module:
    """
    Main training function - orchestrates full training pipeline.
    
    Role in Pipeline:
        Entry point for training - reads config, sets up all components,
        runs training loop, and saves the trained model.
    
    Connections:
        - Uses: utils.py (Config loading)
        - Uses: data.py (Dataset creation)
        - Uses: model.py (Model creation)
        - Output to: Saved model file for evaluation.py
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Trained model
    
    Example:
        >>> model = train_model('configs/default.yaml')
    """
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Import here to avoid circular imports
    from .data import ProteinDataset, ProteinTokenizer
    from .model import create_model
    
    # Create tokenizer and dataset
    tokenizer = ProteinTokenizer()
    
    # Load data (placeholder - adjust paths as needed)
    train_data_path = config.get('data', {}).get('train_data_path', 'data/processed/train_data.npy')
    
    # For demo, create dummy data if file doesn't exist
    if not Path(train_data_path).exists():
        print(f"Warning: {train_data_path} not found. Using dummy data.")
        sequences = ["ACDEFGHIKLMNPQRSTVWY" * 10] * 100
        labels = [0] * 50 + [1] * 50
    else:
        data = np.load(train_data_path, allow_pickle=True).item()
        sequences = data['sequences']
        labels = data['labels']
    
    dataset = ProteinDataset(
        sequences=sequences,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config.get('data', {}).get('max_sequence_length', 512)
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config.get('training', {}).get('batch_size', 256), 
        shuffle=True
    )

    # Initialize model
    model = create_model(config)
    model = model.to(device)
    
    # Setup optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config.get('training', {}).get('learning_rate', 0.001),
        weight_decay=config.get('training', {}).get('weight_decay', 1e-5)
    )
    
    # Setup scheduler
    scheduler_config = config.get('training', {}).get('scheduler', {})
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_config.get('step_size', 10),
        gamma=scheduler_config.get('gamma', 0.1)
    )

    # Loss function
    criterion = ContrastiveLoss()

    # Training loop
    model.train()
    epochs = config.get('training', {}).get('epochs', 50)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                padding_mask = (attention_mask == 0).to(device)
            else:
                padding_mask = None
            
            outputs = model(inputs, src_key_padding_mask=padding_mask)
            
            # For contrastive loss, need pairs - simplified here
            # In practice, you'd sample positive/negative pairs
            loss = outputs.mean()  # Placeholder
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    # Run training from command line
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/default.yaml'
    train_model(config_path)
