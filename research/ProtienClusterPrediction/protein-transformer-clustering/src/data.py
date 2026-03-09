"""
Data Pipeline for Protein Transformer Clustering.

This module consolidates all data handling components:
- ProteinTokenizer: Converts amino acid sequences to token IDs
- FASTALoader: Reads and parses FASTA sequence files
- ProteinDataset: PyTorch Dataset for model training/inference

Data Flow:
    FASTA Files → FASTALoader → Tokenizer → ProteinDataset → DataLoader → Model

Connections to Other Modules:
    - Output to: model.py (via DataLoader batches)
    - Uses: utils.py (for config and file I/O)
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

# ============================================================
# SECTION: PROTEIN TOKENIZER
# Purpose: Convert amino acid sequences to integer token IDs
# Dependencies: None (base component)
# Used by: ProteinDataset
# ============================================================

class ProteinTokenizer:
    """
    Tokenizer for protein/amino acid sequences.
    
    Role in Pipeline:
        Converts raw amino acid sequences (strings like 'ACDEFG') into
        integer token IDs that the neural network can process. Also handles
        special tokens and provides decoding back to sequences.
    
    Connections:
        - Input from: FASTALoader (raw sequence strings)
        - Output to: ProteinDataset (integer token IDs)
    
    Vocabulary:
        Default vocabulary includes 20 standard amino acids plus optional
        special tokens (PAD, UNK, CLS, SEP) for sequence handling.
    
    Args:
        vocab: Optional custom vocabulary list. If None, uses standard amino acids.
        add_special_tokens: Whether to include PAD, UNK, CLS, SEP tokens
    
    Example:
        >>> tokenizer = ProteinTokenizer()
        >>> tokens = tokenizer.tokenize("ACDEFG")
        >>> sequence = tokenizer.detokenize(tokens)
    """
    
    # Standard 20 amino acids (single letter codes)
    STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    # Special token definitions
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    CLS_TOKEN = "<CLS>"
    SEP_TOKEN = "<SEP>"
    
    def __init__(
        self, 
        vocab: Optional[List[str]] = None,
        add_special_tokens: bool = True
    ):
        # Build vocabulary
        if vocab is None:
            self.vocab = self._build_default_vocab(add_special_tokens)
        else:
            self.vocab = vocab
        
        # Create bidirectional mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Store special token IDs for easy access
        self.pad_token_id = self.token_to_id.get(self.PAD_TOKEN, 0)
        self.unk_token_id = self.token_to_id.get(self.UNK_TOKEN, 0)
        self.cls_token_id = self.token_to_id.get(self.CLS_TOKEN)
        self.sep_token_id = self.token_to_id.get(self.SEP_TOKEN)
    
    def _build_default_vocab(self, add_special_tokens: bool) -> List[str]:
        """
        Build the default vocabulary.
        
        Args:
            add_special_tokens: Include PAD, UNK, CLS, SEP tokens
        
        Returns:
            List of vocabulary tokens
        """
        vocab = []
        
        if add_special_tokens:
            # Special tokens first (index 0-3)
            vocab.extend([self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN])
        
        # Standard amino acids
        vocab.extend(list(self.STANDARD_AMINO_ACIDS))
        
        return vocab
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)
    
    def tokenize(
        self, 
        sequence: str,
        add_special_tokens: bool = False
    ) -> List[int]:
        """
        Convert a sequence string to token IDs.
        
        Args:
            sequence: Amino acid sequence string (e.g., "ACDEFG")
            add_special_tokens: Add CLS at start and SEP at end
        
        Returns:
            List of integer token IDs
        """
        # Convert each character, using UNK for unknown amino acids
        tokens = [
            self.token_to_id.get(aa, self.unk_token_id) 
            for aa in sequence.upper()
            if aa in self.token_to_id  # Skip completely unknown chars
        ]
        
        # Add special tokens if requested
        if add_special_tokens and self.cls_token_id is not None:
            tokens = [self.cls_token_id] + tokens + [self.sep_token_id]
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to a sequence string.
        
        Args:
            token_ids: List of integer token IDs
        
        Returns:
            Amino acid sequence string
        """
        return ''.join(
            self.id_to_token.get(idx, '') 
            for idx in token_ids 
            if idx in self.id_to_token and self.id_to_token[idx] not in 
               [self.PAD_TOKEN, self.UNK_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN]
        )
    
    def encode(
        self, 
        sequence: str,
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True
    ) -> List[int]:
        """
        Full encoding with padding and truncation (for model input).
        
        Args:
            sequence: Amino acid sequence string
            max_length: Maximum sequence length
            padding: Padding strategy - 'max_length' or None
            truncation: Whether to truncate sequences longer than max_length
        
        Returns:
            List of token IDs with padding/truncation applied
        """
        tokens = self.tokenize(sequence)
        
        # Truncate if needed
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad if needed
        if padding == 'max_length' and max_length:
            pad_length = max_length - len(tokens)
            tokens = tokens + [self.pad_token_id] * pad_length
        
        return tokens
    
    def decode(self, token_ids_list: List[List[int]]) -> List[str]:
        """
        Batch decode multiple sequences.
        
        Args:
            token_ids_list: List of token ID lists
        
        Returns:
            List of decoded sequence strings
        """
        return [self.detokenize(token_ids) for token_ids in token_ids_list]


# ============================================================
# SECTION: FASTA FILE LOADER
# Purpose: Read and parse FASTA sequence files
# Dependencies: None (file I/O)
# Used by: ProteinDataset, data loading scripts
# ============================================================

def load_fasta_sequences(fasta_file: str) -> List[Tuple[str, str]]:
    """
    Load sequences from a single FASTA file.
    
    Role in Pipeline:
        First step in data loading - reads raw FASTA files and extracts
        sequence headers and sequences for downstream processing.
    
    FASTA Format:
        >header_line
        SEQUENCE_DATA_LINE_1
        SEQUENCE_DATA_LINE_2
        ...
    
    Connections:
        - Input from: Raw FASTA files in data/raw/
        - Output to: ProteinDataset or further processing
    
    Args:
        fasta_file: Path to FASTA file
    
    Returns:
        List of (header, sequence) tuples
    
    Example:
        >>> sequences = load_fasta_sequences("data/raw/proteins.fasta")
        >>> for header, seq in sequences:
        ...     print(f"{header}: {len(seq)} residues")
    """
    sequences = []
    current_header = None
    current_sequence = ""
    
    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None and current_sequence:
                    sequences.append((current_header, current_sequence))
                
                # Start new sequence
                current_header = line[1:]  # Remove '>'
                current_sequence = ""
            else:
                # Append to current sequence
                current_sequence += line
        
        # Don't forget the last sequence
        if current_header is not None and current_sequence:
            sequences.append((current_header, current_sequence))
    
    return sequences


def load_fasta_sequences_only(fasta_file: str) -> List[str]:
    """
    Load only sequences (no headers) from a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
    
    Returns:
        List of sequence strings
    """
    return [seq for _, seq in load_fasta_sequences(fasta_file)]


def load_all_fasta_from_directory(
    directory: str,
    extension: str = ".fasta"
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load all FASTA files from a directory.
    
    Role in Pipeline:
        Batch loading - reads multiple FASTA files at once for
        training on large datasets across multiple files.
    
    Connections:
        - Input from: Directory of FASTA files
        - Output to: Data preprocessing or ProteinDataset
    
    Args:
        directory: Path to directory containing FASTA files
        extension: File extension to filter by (default: .fasta)
    
    Returns:
        Dictionary mapping filename to list of (header, sequence) tuples
    
    Example:
        >>> all_data = load_all_fasta_from_directory("data/raw/")
        >>> for filename, sequences in all_data.items():
        ...     print(f"{filename}: {len(sequences)} sequences")
    """
    all_sequences = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            all_sequences[filename] = load_fasta_sequences(file_path)
    
    return all_sequences


# ============================================================
# SECTION: PROTEIN DATASET
# Purpose: PyTorch Dataset for training and inference
# Dependencies: ProteinTokenizer, FASTALoader functions
# Used by: training.py (DataLoader wraps this)
# ============================================================

class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein sequences.
    
    Role in Pipeline:
        Core data container - wraps tokenized sequences and labels in a
        format compatible with PyTorch DataLoader for batched training.
    
    Connections:
        - Input from: FASTALoader (sequences), ProteinTokenizer (tokenization)
        - Output to: DataLoader → Model (in training.py or evaluation.py)
    
    Args:
        sequences: List of amino acid sequence strings
        labels: List of labels (cluster IDs, class labels, etc.)
        tokenizer: ProteinTokenizer instance for encoding
        max_length: Maximum sequence length (for padding/truncation)
    
    Example:
        >>> tokenizer = ProteinTokenizer()
        >>> dataset = ProteinDataset(sequences, labels, tokenizer)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     model(batch['input_ids'])
    """
    def __init__(
        self, 
        sequences: List[str], 
        labels: List[int],
        tokenizer: ProteinTokenizer,
        max_length: int = 512
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        assert len(sequences) == len(labels), \
            f"Sequences ({len(sequences)}) and labels ({len(labels)}) must match"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary containing:
                - 'input_ids': Tokenized sequence tensor
                - 'label': Label tensor
                - 'attention_mask': Mask for padding tokens
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize and pad/truncate the sequence
        tokenized_sequence = self.tokenizer.encode(
            sequence, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [
            1 if token != self.tokenizer.pad_token_id else 0 
            for token in tokenized_sequence
        ]

        return {
            'input_ids': torch.tensor(tokenized_sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


class UnlabeledProteinDataset(Dataset):
    """
    PyTorch Dataset for unlabeled protein sequences (for inference/clustering).
    
    Role in Pipeline:
        Used during inference when cluster labels are not yet assigned.
        Provides sequences for embedding extraction before clustering.
    
    Connections:
        - Input from: FASTALoader, ProteinTokenizer
        - Output to: Model for embedding extraction → Clustering
    
    Args:
        sequences: List of amino acid sequence strings
        tokenizer: ProteinTokenizer instance
        max_length: Maximum sequence length
        headers: Optional sequence identifiers (from FASTA headers)
    """
    def __init__(
        self, 
        sequences: List[str],
        tokenizer: ProteinTokenizer,
        max_length: int = 512,
        headers: Optional[List[str]] = None
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.headers = headers or [f"seq_{i}" for i in range(len(sequences))]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'header'
        """
        sequence = self.sequences[idx]
        
        tokenized = self.tokenizer.encode(
            sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        attention_mask = [
            1 if token != self.tokenizer.pad_token_id else 0
            for token in tokenized
        ]
        
        return {
            'input_ids': torch.tensor(tokenized, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'header': self.headers[idx]
        }


# ============================================================
# SECTION: DATA UTILITIES
# Purpose: Helper functions for data processing
# Dependencies: ProteinTokenizer, ProteinDataset
# Used by: training.py, scripts
# ============================================================

def create_dataset_from_fasta(
    fasta_path: str,
    labels: Optional[List[int]] = None,
    tokenizer: Optional[ProteinTokenizer] = None,
    max_length: int = 512
) -> Union[ProteinDataset, UnlabeledProteinDataset]:
    """
    Convenience function to create a dataset from a FASTA file.
    
    Role in Pipeline:
        Quick dataset creation - combines FASTALoader and Dataset creation
        into a single function call for common use cases.
    
    Args:
        fasta_path: Path to FASTA file
        labels: Optional labels (if None, creates UnlabeledProteinDataset)
        tokenizer: Tokenizer instance (creates one if None)
        max_length: Maximum sequence length
    
    Returns:
        ProteinDataset if labels provided, else UnlabeledProteinDataset
    
    Example:
        >>> dataset = create_dataset_from_fasta("proteins.fasta", labels=[0, 1, 0])
        >>> # Or for clustering:
        >>> dataset = create_dataset_from_fasta("proteins.fasta")  # No labels
    """
    # Load sequences
    sequences_with_headers = load_fasta_sequences(fasta_path)
    headers = [h for h, _ in sequences_with_headers]
    sequences = [s for _, s in sequences_with_headers]
    
    # Create tokenizer if needed
    if tokenizer is None:
        tokenizer = ProteinTokenizer()
    
    # Create appropriate dataset
    if labels is not None:
        return ProteinDataset(sequences, labels, tokenizer, max_length)
    else:
        return UnlabeledProteinDataset(sequences, tokenizer, max_length, headers)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    
    Handles batching of dictionary items from ProteinDataset.
    Use with DataLoader(dataset, collate_fn=collate_fn).
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with stacked tensors
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    
    # Add labels if present
    if 'label' in batch[0]:
        result['label'] = torch.stack([item['label'] for item in batch])
    
    # Add headers if present (for tracking during inference)
    if 'header' in batch[0]:
        result['header'] = [item['header'] for item in batch]
    
    return result
