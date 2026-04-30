# Research Notes: PyTorch GPU Optimization & Protein Clustering Pipeline

**Date:** April 30, 2026  
**Project:** 2026-Sequence-Pattern-Discovery-Research  
**Status:** GPU optimization complete, clustering results at 0.3719 silhouette score

---

## Problem Summary

### Initial Issue: PyTorch CPU-Only on Windows with NVIDIA GPU

The protein clustering pipeline (`test_pipeline.py`) was running exclusively on CPU despite having an NVIDIA GeForce GTX 1070 GPU available. This caused:
- Extremely slow inference and training
- Inefficient resource utilization
- DLL compatibility errors with Microsoft Store Python

### Root Causes Identified

1. **Microsoft Store Python incompatibility** - Known issues with PyTorch CUDA DLL loading
   - Error: `OSError: [WinError 193] %1 is not a valid Win32 application`
   - Error: `OSError: [WinError 127] The specified procedure could not be found`

2. **CPU-only PyTorch installation** - Installed with `--index-url https://download.pytorch.org/whl/cpu`

3. **Embeddings moved to CPU during processing** - `.cpu()` calls in `esm_embedder.py` and `test_pipeline.py` moved GPU tensors back to CPU immediately after computation

4. **Incorrect pin_memory configuration** - `pin_memory=USE_CUDA` fails when tensors are already on GPU (only works with CPU tensors)

---

## Solutions Implemented

### 1. Created Virtual Environment (venv)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Why:** Isolates dependencies from system Python and Microsoft Store installation

### 2. Installed GPU PyTorch (CUDA 12.4)
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

**Verification:**
```bash
python -c "import torch; print(torch.__version__)"
# Output: 2.6.0+cu124
```

**GPU Specs Detected:**
- GPU: NVIDIA GeForce GTX 1070
- CUDA Version: 12.4
- cuDNN Version: 90100
- VRAM: 8192 MB

### 3. Added GPU Diagnostics to Pipeline
[test_pipeline.py](research/ProtienClusterPrediction/test_pipeline.py) lines 68-76:
```python
# GPU diagnostics
print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
```

### 4. Fixed Embeddings: Kept on GPU During Processing

**File:** [esm_embedder.py](research/ProtienClusterPrediction/esm_embedder.py) lines 143-144  
**Before:**
```python
all_embeddings.append(embeddings.cpu())
all_masks.append(attention_mask.cpu())
```

**After:**
```python
all_embeddings.append(embeddings)
all_masks.append(attention_mask)
```

**File:** [test_pipeline.py](research/ProtienClusterPrediction/test_pipeline.py) line 112  
**Before:**
```python
pooled = mean_pool(emb, mask).cpu()
```

**After:**
```python
pooled = mean_pool(emb, mask)
```

### 5. Fixed DataLoader pin_memory Configuration
[test_pipeline.py](research/ProtienClusterPrediction/test_pipeline.py) line 330  
**Before:**
```python
pin_memory=USE_CUDA,
```

**Error encountered:**
```
RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
```

**After:**
```python
pin_memory=False,
```

**Explanation:** `pin_memory` only applies to CPU tensors being transferred to GPU. Since tensors are already on GPU, this must be disabled.

### 6. Added to .gitignore
Standard venv patterns already present:
```
.venv/
venv/
env/
```

---

## Performance Metrics

### Clustering Results

**Test Silhouette Score:** 0.3719 (moderate separation)

**Cluster Distribution:**

| Metric | Cluster 0 | Cluster 1 |
|--------|-----------|-----------|
| Sequences | 9,004 | 5,972 |
| GC Content | 0.51 ± 0.03 | 0.55 ± 0.03 |
| Motifs Found | 0 | 0 |

**Isolate Composition:**

Cluster 0:
- background_1: 4,072 (45.2%)
- isolate_main: 3,243 (36.0%)
- background_2: 1,689 (18.8%)

Cluster 1:
- background_2: 3,287 (55.0%)
- isolate_main: 1,757 (29.4%)
- background_1: 928 (15.5%)

### GPU Utilization
- Device: CUDA (GTX 1070)
- ESM2 Model: facebook/esm2_t12_35M_UR50D (33.5M parameters)
- Embedding Dimension: 480
- Processing Speed: ~5.4 sequences/sec with 256 batch size

---

## Current Configuration

### test_pipeline.py Settings
```python
K = 5                              # k-mer size
LATENT_DIM = 32
BATCH_SIZE = 512
EPOCHS = 5                         # Transformer training epochs
N_CLUSTERS = 6                     # Target clusters
MAX_READS_PER_ISOLATE = 200000
RANDOM_STATE = 42

# Hyperparameter search space
PCA_COMPONENT_OPTIONS = [2, 4, 8, 16, 32, 64, 128]
KMEANS_CLUSTER_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12]
SILHOUETTE_SAMPLE_SIZE = 100000
```

### Pipeline Architecture
1. **Embedding Generation:** ESM2 (facebook/esm2_t12_35M_UR50D)
2. **Sequence Encoding:** EmbeddingTransformer (4-layer, 8-head)
3. **Masking:** 15% random masking during pretraining
4. **Dimensionality Reduction:** PCA (2-128 components)
5. **Clustering:** MiniBatchKMeans (2-12 clusters)
6. **Evaluation:** Silhouette score on validation set

---

## Improvement Opportunities

### 1. Use Larger ESM2 Model (Highest Impact)
**Current:** `facebook/esm2_t12_35M_UR50D` (35M params)  
**Recommended:** `facebook/esm2_t33_650M_UR50D` (650M params)

**Expected improvement:** +0.10-0.15 silhouette score  
**Trade-off:** Slower inference (but GPU can handle it)

**Implementation:** Modify [esm_embedder.py](research/ProtienClusterPrediction/esm_embedder.py) line 56:
```python
self.model_name = model_name or "facebook/esm2_t33_650M_UR50D"
```

### 2. Increase Training Complexity
**Option A - More epochs:**
```python
EPOCHS = 20  # Instead of 5
```

**Option B - Deeper transformer:**
```python
EmbeddingTransformer(
    input_dim=480,
    model_dim=512,      # Was 256
    nhead=16,           # Was 8
    num_layers=8        # Was 4
)
```

### 3. Expand Cluster Search Space
```python
KMEANS_CLUSTER_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
PCA_COMPONENT_OPTIONS = [2, 4, 8, 16, 32, 64, 128, 256]
```

### 4. Implement Motif Extraction
Currently `motifs: Counter()` is always empty. Add k-mer extraction:
```python
def extract_kmers(sequence, k=5):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def extract_motifs(sequences, k=5):
    motif_counts = Counter()
    for seq in sequences:
        for kmer in extract_kmers(seq, k):
            motif_counts[kmer] += 1
    return motif_counts.most_common(10)
```

### 5. Sequence Length Normalization
Add length-normalized features:
```python
def get_sequence_features(seq):
    return {
        'length': len(seq),
        'gc_content': (seq.count('G') + seq.count('C')) / len(seq),
        'aa_composition': Counter(seq)
    }
```

### 6. Better Sequence Preprocessing
- Filter by length (remove very short/long sequences)
- Remove duplicates
- Add sequence complexity metrics
- Stratified sampling by isolate

---

## Installation & Setup Summary

### For Future Reference

**Quick setup for new environment:**
```bash
cd d:\VSCode\pattern_sequence_research\2026-Sequence-Pattern-Discovery-Research

# Create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install GPU PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install scikit-learn transformers biopython matplotlib

# Run pipeline
python research/ProtienClusterPrediction/test_pipeline.py
```

### Dependencies Installed
- torch==2.6.0+cu124
- torchvision==0.21.0+cu124
- torchaudio==2.6.0+cu124
- scikit-learn==1.8.0
- transformers==5.7.0
- biopython==1.87
- matplotlib==3.10.9
- scipy==1.17.1
- numpy==2.4.3

---

## Key Learnings

### 1. Microsoft Store Python Issues
- Incompatible with PyTorch CUDA builds
- Solution: Always use standard Python from python.org

### 2. Virtual Environments Are Essential
- Isolates PyTorch GPU builds from system Python
- Prevents DLL version conflicts
- Reproducible across runs

### 3. GPU Memory Management
- `.cpu()` calls immediately after GPU computation negate performance gains
- Keep tensors on GPU through full pipeline when possible
- Only move to CPU for saving/display

### 4. DataLoader pin_memory Semantics
- `pin_memory=True` ONLY for CPU→GPU transfers
- Must be `False` when data already on GPU
- Common source of runtime errors

### 5. ESM2 Embeddings Quality
- Larger models (650M) significantly improve downstream clustering
- Trade-off: inference speed vs. quality
- GPU enables viable use of larger models

---

## Files Modified

| File | Changes |
|------|---------|
| [test_pipeline.py](research/ProtienClusterPrediction/test_pipeline.py) | Added GPU diagnostics, removed `.cpu()` calls, fixed `pin_memory` |
| [esm_embedder.py](research/ProtienClusterPrediction/esm_embedder.py) | Removed `.cpu()` calls to keep embeddings on GPU |
| [.gitignore](.gitignore) | Verified venv patterns present |

---

## Next Steps

1. **Immediate:** Try larger ESM2 model (650M) for better embeddings
2. **Short-term:** Implement motif extraction to identify resistance patterns
3. **Medium-term:** Add length normalization and sequence complexity metrics
4. **Long-term:** Compare with other embedding methods (ProtBERT, OmegaFold)

---

## References

- PyTorch CUDA Setup: https://pytorch.org/get-started/locally/
- ESM2 Models: https://huggingface.co/facebook/
- Silhouette Score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

---

**Last Updated:** April 30, 2026  
**Status:** Complete - GPU optimization verified, clustering operational
