"""
ESM2 Embedder Template

Simple, plug-and-play wrapper for ESM2 protein embeddings.
No caching, no complexity - just clean ESM2 embedding computation.

Usage:
    from esm_embedder import ESMEmbedder

    embedder = ESMEmbedder()
    embeddings, masks = embedder.embed_sequences(["MKTAY", "AIVLGG"])

    print(embeddings.shape)  # (2, 512, 480) for 35M model
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import config


class ESMEmbedder:
    """
    Simple ESM2 embedding wrapper.

    Features:
    - Automatically downloads ESM2 from Hugging Face
    - Computes per-residue embeddings for protein sequences
    - Handles batching for efficient GPU usage
    - Returns padded embeddings and attention masks
    """

    def __init__(self,
                 model_name: str = None,
                 device: torch.device = None,
                 batch_size: int = None):
        """
        Initialize ESM2 embedder.

        Args:
            model_name: ESM2 model name (default: from config)
            device: Device to run on (default: auto-detect GPU/CPU)
            batch_size: Batch size for processing (default: from config)
        """
        self.model_name = model_name or config.ESM_MODEL_NAME
        self.device = device or config.DEVICE
        self.batch_size = batch_size or config.BATCH_SIZE

        print(f"[ESM2] Initializing embedder:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load ESM2 model and tokenizer from Hugging Face."""
        print(f"[*] Loading ESM2 model from Hugging Face...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Move to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size

        print(f" Model loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Embedding dim: {self.embedding_dim}")

    def embed_sequences(self,
                       sequences: List[str],
                       max_length: int = None,
                       show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ESM2 embeddings for a list of protein sequences.

        Args:
            sequences: List of protein sequences (amino acid strings)
            max_length: Maximum sequence length (default: from config)
            show_progress: Show progress bar (default: True)

        Returns:
            embeddings: Tensor of shape (num_sequences, max_length, embedding_dim)
            masks: Attention masks of shape (num_sequences, max_length)
                   - True = real amino acid, False = padding

        Example:
            >>> embedder = ESMEmbedder()
            >>> seqs = ["MKTAYIAKQRQISFVK", "AIVLGG"]
            >>> embeddings, masks = embedder.embed_sequences(seqs)
            >>> print(embeddings.shape)  # (2, 512, 480)
        """
        max_length = max_length or config.MAX_SEQUENCE_LENGTH

        print(f"[*] Computing embeddings for {len(sequences)} sequences...")

        all_embeddings = []
        all_masks = []

        # Process in batches
        num_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(sequences), self.batch_size)

        if show_progress and num_batches > 1:
            iterator = tqdm(iterator, desc="Computing embeddings", total=num_batches)

        for i in iterator:
            batch_seqs = sequences[i:i + self.batch_size]

            # Tokenize sequences
            tokenized = self.tokenizer(
                batch_seqs,
                add_special_tokens=True,      # Add <cls> and <eos> tokens
                padding='max_length',          # Pad all to same length
                truncation=True,               # Truncate if too long
                max_length=max_length,
                return_tensors="pt"            # Return PyTorch tensors
            )

            # Move to device
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)

            # Compute embeddings (no gradients needed)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # (batch, seq_len, embed_dim)

            # Remove special tokens (<cls> at start, <eos> at end)
            embeddings = embeddings[:, 1:-1, :]
            attention_mask = attention_mask[:, 1:-1]

            # Keep on GPU for faster processing
            all_embeddings.append(embeddings)
            all_masks.append(attention_mask)

        # Concatenate all batches
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        print(f"  Embeddings computed!")
        print(f"  Shape: {embeddings_tensor.shape}")
        print(f"  Device: {embeddings_tensor.device}")

        return embeddings_tensor, masks_tensor

    def embed_single_sequence(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method to embed a single sequence.

        Args:
            sequence: Single protein sequence string

        Returns:
            embedding: Tensor of shape (max_length, embedding_dim)
            mask: Attention mask of shape (max_length,)

        Example:
            >>> embedder = ESMEmbedder()
            >>> emb, mask = embedder.embed_single_sequence("MKTAY")
            >>> print(emb.shape)  # (512, 480)
        """
        embeddings, masks = self.embed_sequences([sequence], show_progress=False)
        return embeddings[0], masks[0]


# ====================================================================
# CONVENIENCE FUNCTION
# ====================================================================

def get_embeddings(sequences: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quick one-liner to get ESM2 embeddings.

    Args:
        sequences: List of protein sequences
        **kwargs: Additional arguments (model_name, device, max_length, etc.)

    Returns:
        embeddings: Tensor of shape (num_sequences, max_length, embedding_dim)
        masks: Attention masks

    Example:
        >>> from esm_embedder import get_embeddings
        >>> embeddings, masks = get_embeddings(["MKTAY", "AIVLGG"])
    """
    embedder = ESMEmbedder()
    return embedder.embed_sequences(sequences, **kwargs)


if __name__ == "__main__":
    # Test the embedder
    print("Testing ESMEmbedder...")

    test_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQ",
        "AIVLGG",
        "SEQWENCE"
    ]

    embedder = ESMEmbedder()
    embeddings, masks = embedder.embed_sequences(test_sequences)

    print(f"\nTest successful!")
    print(f"  Input: {len(test_sequences)} sequences")
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Embedding dim: {embeddings.shape[-1]}")
