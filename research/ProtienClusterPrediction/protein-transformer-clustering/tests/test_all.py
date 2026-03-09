"""
Combined Test Suite for Protein Transformer Clustering.

This module consolidates all tests into a single file:
- Model tests: Transformer architecture validation
- Tokenizer tests: Sequence tokenization/detokenization
- Data tests: Dataset loading and processing
- Integration tests: End-to-end pipeline validation

Run with: pytest tests/test_all.py -v

Structure:
    1. Model Tests - Test neural network components
    2. Tokenizer Tests - Test sequence encoding/decoding
    3. Data Tests - Test data loading pipeline
    4. Integration Tests - Test component interactions
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


# ============================================================
# SECTION: MODEL TESTS
# Purpose: Validate neural network components
# Tests: Initialization, forward pass, training step
# ============================================================

class TestTransformerModel:
    """Tests for the Transformer model architecture."""
    
    def test_positional_encoding_initialization(self):
        """Test PositionalEncoding module initialization."""
        from src.model import PositionalEncoding
        
        pe = PositionalEncoding(embed_dim=256, dropout=0.1, max_len=1000)
        assert pe is not None
        assert hasattr(pe, 'pe')
        assert pe.pe.shape == (1, 1000, 256)
    
    def test_positional_encoding_forward(self):
        """Test PositionalEncoding forward pass."""
        from src.model import PositionalEncoding
        
        pe = PositionalEncoding(embed_dim=128, dropout=0.0)
        x = torch.randn(32, 100, 128)  # (batch, seq_len, embed_dim)
        
        output = pe(x)
        
        assert output.shape == x.shape
        # With dropout=0, output should differ from input (added positional info)
        assert not torch.allclose(output, x)
    
    def test_protein_embedding_initialization(self):
        """Test ProteinEmbedding initialization."""
        from src.model import ProteinEmbedding
        
        embed = ProteinEmbedding(vocab_size=25, embedding_dim=256)
        assert embed is not None
        assert embed.embedding_dim == 256
    
    def test_protein_embedding_forward(self):
        """Test ProteinEmbedding forward pass."""
        from src.model import ProteinEmbedding
        
        embed = ProteinEmbedding(vocab_size=25, embedding_dim=128)
        tokens = torch.randint(0, 25, (16, 50))  # (batch, seq_len)
        
        output = embed(tokens)
        
        assert output.shape == (16, 50, 128)
    
    def test_projection_layer_initialization(self):
        """Test ProjectionLayer initialization."""
        from src.model import ProjectionLayer
        
        proj = ProjectionLayer(input_dim=256, output_dim=64)
        assert proj is not None
    
    def test_projection_layer_forward(self):
        """Test ProjectionLayer forward pass."""
        from src.model import ProjectionLayer
        
        proj = ProjectionLayer(input_dim=128, output_dim=32)
        x = torch.randn(16, 128)
        
        output = proj(x)
        
        assert output.shape == (16, 32)
    
    def test_transformer_initialization(self):
        """Test full Transformer model initialization."""
        from src.model import Transformer
        
        model = Transformer(
            input_dim=25,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            output_dim=64,
            dropout=0.1
        )
        
        assert model is not None
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'transformer_encoder')
        assert hasattr(model, 'fc_out')
    
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass dimensions."""
        from src.model import Transformer
        
        model = Transformer(
            input_dim=25,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            output_dim=64
        )
        
        tokens = torch.randint(0, 25, (8, 100))  # (batch, seq_len)
        output = model(tokens)
        
        assert output.shape == (8, 100, 64)  # (batch, seq_len, output_dim)
    
    def test_transformer_with_padding_mask(self):
        """Test Transformer with attention padding mask."""
        from src.model import Transformer
        
        model = Transformer(
            input_dim=25,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            output_dim=64
        )
        
        batch_size, seq_len = 8, 100
        tokens = torch.randint(0, 25, (batch_size, seq_len))
        
        # Create padding mask (True = ignore token)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        padding_mask[:, -20:] = True  # Last 20 tokens are padding
        
        output = model(tokens, src_key_padding_mask=padding_mask)
        
        assert output.shape == (batch_size, seq_len, 64)
    
    def test_transformer_sequence_embedding(self):
        """Test get_sequence_embedding pooling."""
        from src.model import Transformer
        
        model = Transformer(
            input_dim=25,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            output_dim=64
        )
        
        tokens = torch.randint(0, 25, (8, 100))
        
        # Test mean pooling
        embedding_mean = model.get_sequence_embedding(tokens, pooling='mean')
        assert embedding_mean.shape == (8, 64)
        
        # Test max pooling
        embedding_max = model.get_sequence_embedding(tokens, pooling='max')
        assert embedding_max.shape == (8, 64)
        
        # Test CLS pooling
        embedding_cls = model.get_sequence_embedding(tokens, pooling='cls')
        assert embedding_cls.shape == (8, 64)
    
    def test_transformer_training_step(self):
        """Test single training step (gradient computation)."""
        from src.model import Transformer
        
        model = Transformer(
            input_dim=25,
            embed_dim=64,
            num_heads=2,
            num_layers=1,
            output_dim=32
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        tokens = torch.randint(0, 25, (4, 50))
        target = torch.randn(4, 50, 32)
        
        model.train()
        optimizer.zero_grad()
        output = model(tokens)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Check gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_create_model_factory(self):
        """Test model creation from config dict."""
        from src.model import create_model
        
        config = {
            'vocab_size': 25,
            'embedding_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'latent_dim': 64,
            'dropout_rate': 0.1
        }
        
        model = create_model(config)
        
        assert model is not None
        tokens = torch.randint(0, 25, (2, 50))
        output = model(tokens)
        assert output.shape == (2, 50, 64)


# ============================================================
# SECTION: TOKENIZER TESTS
# Purpose: Validate sequence tokenization
# Tests: Encoding, decoding, special tokens, edge cases
# ============================================================

class TestProteinTokenizer:
    """Tests for the ProteinTokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer creates proper vocabulary."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        
        assert tokenizer.vocab_size > 0
        assert len(tokenizer.token_to_id) == tokenizer.vocab_size
        assert len(tokenizer.id_to_token) == tokenizer.vocab_size
    
    def test_tokenizer_with_special_tokens(self):
        """Test special tokens are properly included."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer(add_special_tokens=True)
        
        assert tokenizer.pad_token_id is not None
        assert tokenizer.unk_token_id is not None
        assert tokenizer.cls_token_id is not None
        assert tokenizer.sep_token_id is not None
    
    def test_tokenization_basic(self):
        """Test basic sequence tokenization."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # All standard amino acids
        
        tokens = tokenizer.tokenize(sequence)
        
        assert len(tokens) == len(sequence)
        assert all(isinstance(t, int) for t in tokens)
    
    def test_tokenization_with_special_tokens_flag(self):
        """Test tokenization adds special tokens when requested."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer(add_special_tokens=True)
        sequence = "ACDEF"
        
        tokens = tokenizer.tokenize(sequence, add_special_tokens=True)
        
        # Should have CLS + sequence + SEP
        assert len(tokens) == len(sequence) + 2
        assert tokens[0] == tokenizer.cls_token_id
        assert tokens[-1] == tokenizer.sep_token_id
    
    def test_detokenization(self):
        """Test converting tokens back to sequence."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        original = "ACDEFGHIK"
        
        tokens = tokenizer.tokenize(original)
        recovered = tokenizer.detokenize(tokens)
        
        assert recovered == original
    
    def test_encode_with_padding(self):
        """Test encoding with padding to max_length."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequence = "ACDEF"  # 5 amino acids
        max_length = 10
        
        encoded = tokenizer.encode(sequence, max_length=max_length, padding='max_length')
        
        assert len(encoded) == max_length
        # Check padding tokens at end
        assert encoded[-1] == tokenizer.pad_token_id
    
    def test_encode_with_truncation(self):
        """Test encoding with truncation."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids
        max_length = 10
        
        encoded = tokenizer.encode(sequence, max_length=max_length, truncation=True)
        
        assert len(encoded) == max_length
    
    def test_empty_sequence(self):
        """Test tokenizing empty sequence."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        tokens = tokenizer.tokenize("")
        
        assert tokens == []
    
    def test_batch_decode(self):
        """Test batch decoding multiple sequences."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEF", "GHIKL", "MNPQR"]
        
        encoded = [tokenizer.tokenize(s) for s in sequences]
        decoded = tokenizer.decode(encoded)
        
        assert decoded == sequences
    
    @pytest.mark.parametrize("sequence,expected_length", [
        ("ACDEFGHIKLMNPQRSTVWY", 20),  # All standard
        ("", 0),                        # Empty
        ("ACDEF", 5),                   # Short
    ])
    def test_tokenization_lengths(self, sequence, expected_length):
        """Parametrized test for various sequence lengths."""
        from src.data import ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        tokens = tokenizer.tokenize(sequence)
        
        assert len(tokens) == expected_length


# ============================================================
# SECTION: DATA TESTS
# Purpose: Validate data loading and dataset classes
# Tests: FASTA loading, Dataset creation, DataLoader batching
# ============================================================

class TestFASTALoader:
    """Tests for FASTA file loading functions."""
    
    def test_load_fasta_sequences(self, tmp_path):
        """Test loading sequences from FASTA file."""
        from src.data import load_fasta_sequences
        
        # Create temporary FASTA file
        fasta_content = """>seq1_header
ACDEFGHIK
>seq2_header
LMNPQRSTV
"""
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        
        sequences = load_fasta_sequences(str(fasta_file))
        
        assert len(sequences) == 2
        assert sequences[0] == ("seq1_header", "ACDEFGHIK")
        assert sequences[1] == ("seq2_header", "LMNPQRSTV")
    
    def test_load_multiline_fasta(self, tmp_path):
        """Test loading FASTA with multiline sequences."""
        from src.data import load_fasta_sequences
        
        fasta_content = """>seq1
ACDEF
GHIKL
MNPQR
"""
        fasta_file = tmp_path / "multiline.fasta"
        fasta_file.write_text(fasta_content)
        
        sequences = load_fasta_sequences(str(fasta_file))
        
        assert len(sequences) == 1
        assert sequences[0][1] == "ACDEFGHIKLMNPQR"
    
    def test_load_fasta_sequences_only(self, tmp_path):
        """Test loading only sequences (no headers)."""
        from src.data import load_fasta_sequences_only
        
        fasta_content = """>header1
ACDEF
>header2
GHIKL
"""
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content)
        
        sequences = load_fasta_sequences_only(str(fasta_file))
        
        assert sequences == ["ACDEF", "GHIKL"]


class TestProteinDataset:
    """Tests for ProteinDataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset initializes correctly."""
        from src.data import ProteinDataset, ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEF", "GHIKL", "MNPQR"]
        labels = [0, 1, 0]
        
        dataset = ProteinDataset(sequences, labels, tokenizer, max_length=20)
        
        assert len(dataset) == 3
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct structure."""
        from src.data import ProteinDataset, ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEFGHIK"]
        labels = [1]
        
        dataset = ProteinDataset(sequences, labels, tokenizer, max_length=20)
        sample = dataset[0]
        
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'label' in sample
        assert sample['input_ids'].shape == (20,)
        assert sample['attention_mask'].shape == (20,)
        assert sample['label'].item() == 1
    
    def test_dataset_with_dataloader(self):
        """Test dataset works with PyTorch DataLoader."""
        from src.data import ProteinDataset, ProteinTokenizer
        from torch.utils.data import DataLoader
        
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEF"] * 10
        labels = [0] * 10
        
        dataset = ProteinDataset(sequences, labels, tokenizer, max_length=20)
        loader = DataLoader(dataset, batch_size=4)
        
        batch = next(iter(loader))
        
        assert batch['input_ids'].shape == (4, 20)
        assert batch['label'].shape == (4,)


class TestUnlabeledDataset:
    """Tests for UnlabeledProteinDataset."""
    
    def test_unlabeled_dataset(self):
        """Test unlabeled dataset for inference."""
        from src.data import UnlabeledProteinDataset, ProteinTokenizer
        
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEF", "GHIKL"]
        headers = ["protein_1", "protein_2"]
        
        dataset = UnlabeledProteinDataset(sequences, tokenizer, headers=headers)
        sample = dataset[0]
        
        assert 'input_ids' in sample
        assert 'header' in sample
        assert 'label' not in sample
        assert sample['header'] == "protein_1"


# ============================================================
# SECTION: TRAINING TESTS
# Purpose: Validate training components
# Tests: Loss functions, scheduler, trainer
# ============================================================

class TestLossFunctions:
    """Tests for loss function implementations."""
    
    def test_contrastive_loss(self):
        """Test ContrastiveLoss computation."""
        from src.training import ContrastiveLoss
        
        loss_fn = ContrastiveLoss(margin=1.0)
        
        embed1 = torch.randn(8, 64)
        embed2 = torch.randn(8, 64)
        labels = torch.randint(0, 2, (8,)).float()
        
        loss = loss_fn(embed1, embed2, labels)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
    
    def test_clustering_loss(self):
        """Test ClusteringLoss computation."""
        from src.training import ClusteringLoss
        
        loss_fn = ClusteringLoss()
        
        embeddings = torch.randn(16, 64)
        centers = torch.randn(5, 64)  # 5 clusters
        labels = torch.randint(0, 5, (16,))
        
        loss = loss_fn(embeddings, centers, labels)
        
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestScheduler:
    """Tests for learning rate scheduler."""
    
    def test_warmup_scheduler(self):
        """Test WarmupScheduler learning rate progression."""
        from src.training import WarmupScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = WarmupScheduler(
            optimizer, 
            warmup_steps=100, 
            total_steps=1000,
            initial_lr=1e-7,
            final_lr=1e-3
        )
        
        # Initial LR should be low
        initial_lr = scheduler.get_lr()
        
        # After warmup, should reach final_lr
        for _ in range(100):
            scheduler.step()
        
        warmup_lr = scheduler.get_lr()
        
        assert warmup_lr > initial_lr
    
    def test_scheduler_reset(self):
        """Test scheduler reset functionality."""
        from src.training import WarmupScheduler
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        scheduler = WarmupScheduler(
            optimizer, warmup_steps=10, total_steps=100,
            initial_lr=0.001, final_lr=0.01
        )
        
        for _ in range(50):
            scheduler.step()
        
        scheduler.reset()
        
        assert scheduler.current_step == 0


# ============================================================
# SECTION: EVALUATION TESTS  
# Purpose: Validate evaluation metrics and clustering
# Tests: Metrics calculation, cluster analysis
# ============================================================

class TestMetrics:
    """Tests for clustering metrics."""
    
    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        from src.evaluation import calculate_silhouette_score
        
        # Create clearly separated clusters
        embeddings = np.vstack([
            np.random.randn(50, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.random.randn(50, 10) + np.array([-5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ])
        labels = np.array([0] * 50 + [1] * 50)
        
        score = calculate_silhouette_score(embeddings, labels)
        
        assert score is not None
        assert -1 <= score <= 1
        # Well-separated clusters should have high score
        assert score > 0.5
    
    def test_silhouette_single_cluster(self):
        """Test silhouette with single cluster returns None."""
        from src.evaluation import calculate_silhouette_score
        
        embeddings = np.random.randn(50, 10)
        labels = np.zeros(50)  # All same cluster
        
        score = calculate_silhouette_score(embeddings, labels)
        
        assert score is None
    
    def test_adjusted_rand_index(self):
        """Test adjusted rand index calculation."""
        from src.evaluation import calculate_adjusted_rand_index
        
        true = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([0, 0, 0, 1, 1, 1])  # Perfect match
        
        ari = calculate_adjusted_rand_index(true, pred)
        
        assert ari == 1.0
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        from src.evaluation import calculate_all_metrics
        
        embeddings = np.random.randn(100, 10)
        predicted = np.random.randint(0, 5, 100)
        true = np.random.randint(0, 5, 100)
        
        metrics = calculate_all_metrics(embeddings, predicted, true)
        
        assert 'silhouette_score' in metrics
        assert 'adjusted_rand_index' in metrics
        assert 'normalized_mutual_info' in metrics


class TestClusterAnalysis:
    """Tests for cluster analysis utilities."""
    
    def test_cluster_distribution(self):
        """Test getting cluster distribution."""
        from src.evaluation import get_cluster_distribution
        
        labels = np.array([0, 0, 0, 1, 1, 2])
        dist = get_cluster_distribution(labels)
        
        assert dist == {0: 3, 1: 2, 2: 1}
    
    def test_summarize_clusters(self):
        """Test cluster summarization."""
        from src.evaluation import summarize_clusters
        
        labels = np.array([0, 0, 1, 1])
        sequences = ["ACDEF", "GHIKL", "MN", "PQRSTVWY"]
        
        summary = summarize_clusters(labels, sequences=sequences)
        
        assert 0 in summary
        assert 1 in summary
        assert summary[0]['count'] == 2
        assert summary[1]['count'] == 2
        assert 'mean_length' in summary[0]


# ============================================================
# SECTION: UTILITIES TESTS
# Purpose: Validate utility functions
# Tests: Config, I/O, helpers
# ============================================================

class TestConfig:
    """Tests for configuration management."""
    
    def test_config_load(self, tmp_path):
        """Test loading configuration from YAML."""
        from src.utils import Config
        
        config_content = """
model: transformer
embedding_dim: 128
training:
  batch_size: 32
  epochs: 10
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        assert config['model'] == 'transformer'
        assert config['embedding_dim'] == 128
        assert config['training']['batch_size'] == 32
    
    def test_config_get_with_default(self, tmp_path):
        """Test config.get() with default value."""
        from src.utils import Config
        
        config_content = "model: transformer"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_config_file_not_found(self):
        """Test error on missing config file."""
        from src.utils import Config
        
        with pytest.raises(FileNotFoundError):
            Config('nonexistent_config.yaml')


class TestModelIO:
    """Tests for model save/load functions."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading model checkpoint."""
        from src.utils import save_model, load_model
        
        # Create simple model
        model = torch.nn.Linear(10, 5)
        original_weight = model.weight.clone()
        
        filepath = tmp_path / "model.pt"
        save_model(model, str(filepath))
        
        # Create new model and load
        new_model = torch.nn.Linear(10, 5)
        load_model(new_model, str(filepath))
        
        assert torch.allclose(new_model.weight, original_weight)


class TestHelperFunctions:
    """Tests for general utility functions."""
    
    def test_setup_device(self):
        """Test device setup."""
        from src.utils import setup_device
        
        device = setup_device(use_cuda=False)
        assert device == torch.device('cpu')
    
    def test_count_parameters(self):
        """Test parameter counting."""
        from src.utils import count_parameters
        
        model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        count = count_parameters(model)
        
        assert count == 55
    
    def test_format_number(self):
        """Test number formatting."""
        from src.utils import format_number
        
        assert format_number(500) == "500"
        assert format_number(1500) == "1.5K"
        assert format_number(1500000) == "1.5M"
        assert format_number(1500000000) == "1.5B"
    
    def test_ensure_dir(self, tmp_path):
        """Test directory creation."""
        from src.utils import ensure_dir
        
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = ensure_dir(new_dir)
        
        assert result.exists()
        assert result.is_dir()


# ============================================================
# SECTION: INTEGRATION TESTS
# Purpose: Test component interactions
# Tests: End-to-end pipeline validation
# ============================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_data_to_model_pipeline(self):
        """Test data loading through model inference."""
        from src.data import ProteinDataset, ProteinTokenizer
        from src.model import Transformer
        from torch.utils.data import DataLoader
        
        # Create data
        tokenizer = ProteinTokenizer()
        sequences = ["ACDEFGHIK"] * 8
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        dataset = ProteinDataset(sequences, labels, tokenizer, max_length=20)
        loader = DataLoader(dataset, batch_size=4)
        
        # Create model
        model = Transformer(
            input_dim=tokenizer.vocab_size,
            embed_dim=64,
            num_heads=2,
            num_layers=1,
            output_dim=32
        )
        
        # Run inference
        model.eval()
        with torch.no_grad():
            for batch in loader:
                output = model(batch['input_ids'])
                assert output.shape[0] == batch['input_ids'].shape[0]
    
    def test_training_loss_backward(self):
        """Test loss computation and backward pass."""
        from src.model import Transformer
        from src.training import ContrastiveLoss
        
        model = Transformer(
            input_dim=25,
            embed_dim=64,
            num_heads=2,
            num_layers=1,
            output_dim=32
        )
        
        loss_fn = ContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        tokens1 = torch.randint(0, 25, (4, 20))
        tokens2 = torch.randint(0, 25, (4, 20))
        labels = torch.randint(0, 2, (4,)).float()
        
        model.train()
        optimizer.zero_grad()
        
        out1 = model.get_sequence_embedding(tokens1)
        out2 = model.get_sequence_embedding(tokens2)
        
        loss = loss_fn(out1, out2, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() >= 0


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
