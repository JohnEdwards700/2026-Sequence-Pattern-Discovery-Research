"""
Neural Network Components for Protein Transformer Clustering.

This module consolidates all neural network building blocks:
- PositionalEncoding: Adds positional information to embeddings
- ProteinEmbedding: Converts tokenized sequences to dense vectors
- ProjectionLayer: Reduces dimensionality for clustering
- Transformer: Main encoder architecture for sequence understanding

Pipeline Flow:
    Tokens → Embedding → PositionalEncoding → Transformer → Projection → Cluster Space

Connections to Other Modules:
    - Input from: data.py (tokenized sequences via DataLoader)
    - Output to: training.py (loss computation) or evaluation.py (clustering)
"""

import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ============================================================
# SECTION: POSITIONAL ENCODING
# Purpose: Add position information to sequence embeddings
# Dependencies: None (base building block)
# Used by: Transformer, ProteinEmbedding
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Role in Pipeline:
        Injects positional information into embeddings since transformers
        have no inherent notion of sequence order. Uses sine/cosine functions
        of different frequencies to encode positions.
    
    Connections:
        - Input from: ProteinEmbedding output
        - Output to: Transformer encoder layers
    
    Args:
        embed_dim: Dimension of the embedding vectors (must match model dim)
        dropout: Dropout probability applied after adding positions
        max_len: Maximum sequence length to pre-compute encodings for
    
    Example:
        >>> pe = PositionalEncoding(embed_dim=256, dropout=0.1)
        >>> x = torch.randn(32, 100, 256)  # (batch, seq_len, embed_dim)
        >>> output = pe(x)  # Same shape with positional info added
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings: shape (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sine/cosine frequencies
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Embeddings of shape (batch_size, seq_len, embed_dim)
        Returns:
            Position-encoded embeddings of same shape
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# SECTION: PROTEIN EMBEDDING LAYER
# Purpose: Convert discrete protein tokens into continuous embeddings
# Dependencies: None (base building block)
# Used by: Transformer
# ============================================================

class ProteinEmbedding(nn.Module):
    """
    Embedding layer for protein sequences.
    
    Role in Pipeline:
        First layer - converts tokenized amino acid sequences (integer IDs)
        into dense vector representations that the transformer can process.
        Scales embeddings by sqrt(embed_dim) for training stability.
    
    Connections:
        - Input from: Tokenizer (integer token IDs from data.py)
        - Output to: PositionalEncoding → Transformer encoder
    
    Args:
        vocab_size: Number of unique tokens (amino acids + special tokens)
        embedding_dim: Dimension of embedding vectors
    
    Example:
        >>> embed = ProteinEmbedding(vocab_size=25, embedding_dim=256)
        >>> tokens = torch.randint(0, 25, (32, 100))  # (batch, seq_len)
        >>> embeddings = embed(tokens)  # (32, 100, 256)
    """
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(ProteinEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed token IDs to dense vectors.
        
        Args:
            x: Token IDs of shape (batch_size, seq_len)
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(x)


# ============================================================
# SECTION: PROJECTION LAYER
# Purpose: Project transformer outputs to clustering-friendly space
# Dependencies: None (base building block)
# Used by: Transformer (final layer before clustering)
# ============================================================

class ProjectionLayer(nn.Module):
    """
    Linear projection for dimensionality reduction.
    
    Role in Pipeline:
        Final layer - projects high-dimensional transformer outputs
        into a lower-dimensional space suitable for clustering algorithms
        like K-means or DBSCAN.
    
    Connections:
        - Input from: Transformer encoder output (pooled)
        - Output to: Clustering algorithm in evaluation.py
    
    Args:
        input_dim: Dimension of transformer hidden states
        output_dim: Dimension of clustering space (typically 64-256)
    
    Example:
        >>> proj = ProjectionLayer(input_dim=256, output_dim=64)
        >>> hidden = torch.randn(32, 256)  # (batch, hidden_dim)
        >>> projected = proj(hidden)  # (32, 64)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to lower-dimensional space.
        
        Args:
            x: Hidden states of shape (batch_size, input_dim) or 
               (batch_size, seq_len, input_dim)
        Returns:
            Projections of shape (batch_size, output_dim) or 
            (batch_size, seq_len, output_dim)
        """
        return self.linear(x)


# ============================================================
# SECTION: TRANSFORMER ENCODER
# Purpose: Main model architecture for protein sequence encoding
# Dependencies: PositionalEncoding, ProteinEmbedding (can use), ProjectionLayer
# Used by: training.py (train_model), evaluation.py (clustering)
# ============================================================

class Transformer(nn.Module):
    """
    Complete transformer encoder for protein sequence representation.
    
    Role in Pipeline:
        Core model - takes tokenized sequences and produces dense embeddings
        suitable for downstream clustering tasks. Combines embedding,
        positional encoding, transformer layers, and projection.
    
    Connections:
        - Input from: DataLoader (batched tokenized sequences from data.py)
        - Output to: Loss functions (training.py) or Clustering (evaluation.py)
    
    Architecture:
        1. Token Embedding: Convert token IDs to vectors
        2. Positional Encoding: Add sequence position information
        3. Transformer Encoder: Self-attention layers for context
        4. Output Projection: Map to final output dimension
    
    Args:
        input_dim: Vocabulary size (number of unique tokens)
        embed_dim: Dimension of embeddings and transformer hidden size
        num_heads: Number of attention heads (must divide embed_dim)
        num_layers: Number of transformer encoder layers
        output_dim: Final output dimension (projection space)
        dropout: Dropout probability for regularization
    
    Example:
        >>> model = Transformer(
        ...     input_dim=25, embed_dim=256, num_heads=8,
        ...     num_layers=6, output_dim=64
        ... )
        >>> tokens = torch.randint(0, 25, (32, 100))
        >>> output = model(tokens)  # (32, 100, 64)
    """
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_heads: int, 
        num_layers: int, 
        output_dim: int, 
        dropout: float = 0.1
    ):
        super(Transformer, self).__init__()
        
        # Component 1: Token embedding (tokens → vectors)
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.embed_dim = embed_dim
        
        # Component 2: Positional encoding (add position info)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Component 3: Transformer encoder stack (self-attention layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Component 4: Output projection (dimension reduction)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass through the transformer model.
        
        Args:
            x: Token IDs of shape (batch_size, seq_len)
            src_mask: Optional attention mask for causal/custom attention
            src_key_padding_mask: Optional mask for padding tokens (True = ignore)
        
        Returns:
            Output embeddings of shape (batch_size, seq_len, output_dim)
        """
        # Step 1: Embed tokens and scale
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # Step 2: Add positional information
        x = self.positional_encoding(x)
        
        # Step 3: Pass through transformer encoder
        x = self.transformer_encoder(
            x, 
            mask=src_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Step 4: Project to output dimension
        x = self.fc_out(x)
        
        return x
    
    def get_sequence_embedding(
        self, 
        x: torch.Tensor, 
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Get a single embedding vector per sequence (for clustering).
        
        Args:
            x: Token IDs of shape (batch_size, seq_len)
            pooling: Pooling strategy - 'mean', 'max', or 'cls'
        
        Returns:
            Sequence embeddings of shape (batch_size, output_dim)
        """
        # Get full sequence output
        output = self.forward(x)  # (batch, seq_len, output_dim)
        
        # Pool across sequence dimension
        if pooling == 'mean':
            return output.mean(dim=1)
        elif pooling == 'max':
            return output.max(dim=1)[0]
        elif pooling == 'cls':
            return output[:, 0, :]  # First token (CLS-style)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")


# ============================================================
# SECTION: MODEL FACTORY
# Purpose: Create model instances from configuration
# Dependencies: All model classes above
# Used by: training.py, evaluation.py
# ============================================================

def create_model(config: dict) -> Transformer:
    """
    Factory function to create a Transformer model from configuration.
    
    Role in Pipeline:
        Entry point for model creation - reads hyperparameters from
        config dict (loaded from YAML via utils.py) and instantiates model.
    
    Connections:
        - Input from: utils.py (Config class)
        - Output to: training.py (for training) or evaluation.py (for inference)
    
    Args:
        config: Dictionary containing model hyperparameters:
            - embedding_dim: Dimension of embeddings
            - latent_dim: Output/projection dimension
            - num_heads: Number of attention heads
            - num_layers: Number of transformer layers
            - dropout_rate: Dropout probability
            - vocab_size: Size of token vocabulary (optional, default 25)
    
    Returns:
        Initialized Transformer model
    
    Example:
        >>> config = {'embedding_dim': 256, 'latent_dim': 64, ...}
        >>> model = create_model(config)
    """
    return Transformer(
        input_dim=config.get('vocab_size', 25),  # 20 amino acids + special tokens
        embed_dim=config.get('embedding_dim', 128),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        output_dim=config.get('latent_dim', 64),
        dropout=config.get('dropout_rate', 0.1)
    )


# ============================================================
# SECTION: VISUALIZATION FUNCTIONS
# Purpose: Plot embeddings from the transformer model
# ============================================================

def plot_projection(
    model: Transformer,
    tokens: torch.Tensor,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    dim: int = 2,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot embeddings using direct linear projection to 2D/3D.
    
    Args:
        model: Trained Transformer model
        tokens: Token IDs (batch_size, seq_len)
        labels: Optional cluster labels for coloring
        dim: 2 or 3 for output dimensions
        save_path: Optional path to save figure
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.get_sequence_embedding(tokens, pooling='mean')
        # Linear projection to plotting dimensions
        proj = nn.Linear(embeddings.shape[-1], dim)
        coords = proj(embeddings).numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    fig = plt.figure(figsize=(10, 8))
    if dim == 2:
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap='tab10', alpha=0.7)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
    
    plt.title('Projection Plot')
    if labels is not None:
        plt.colorbar(label='Cluster')
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_tsne_pca(
    model: Transformer,
    tokens: torch.Tensor,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    method: str = 'tsne',
    dim: int = 2,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot embeddings using t-SNE or PCA dimensionality reduction.
    
    Args:
        model: Trained Transformer model
        tokens: Token IDs (batch_size, seq_len)
        labels: Optional cluster labels for coloring
        method: 'tsne' or 'pca'
        dim: 2 or 3 for output dimensions
        save_path: Optional path to save figure
    """
    model.eval()
    with torch.no_grad():
        embeddings = model.get_sequence_embedding(tokens, pooling='mean').numpy()
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=dim, random_state=42, perplexity=min(30, len(embeddings)-1))
    else:
        reducer = PCA(n_components=dim)
    coords = reducer.fit_transform(embeddings)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    fig = plt.figure(figsize=(10, 8))
    if dim == 2:
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap='tab10', alpha=0.7)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.set_zlabel(f'{method.upper()} 3')
    
    plt.title(f'{method.upper()} Visualization')
    if labels is not None:
        plt.colorbar(label='Cluster')
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
