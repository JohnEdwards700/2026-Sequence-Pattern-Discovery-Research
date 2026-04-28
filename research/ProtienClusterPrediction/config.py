"""
ESM2 Configuration Template

Simple configuration file for ESM2 protein embeddings.
Modify these settings for your specific use case.
"""

import torch

# ====================================================================
# ESM2 MODEL CONFIGURATION
# ====================================================================

# Choose your ESM2 model size
# Smaller = faster, Larger = better quality embeddings
ESM_MODEL_OPTIONS = {
    "8M": "facebook/esm2_t6_8M_UR50D",      # Fastest (320-dim embeddings)
    "35M": "facebook/esm2_t12_35M_UR50D",   # Good balance (480-dim) ← RECOMMENDED
    "150M": "facebook/esm2_t30_150M_UR50D", # High quality (640-dim)
    "650M": "facebook/esm2_t33_650M_UR50D", # Best quality (1280-dim)
}

# Select which model to use
ESM_MODEL_NAME = ESM_MODEL_OPTIONS["35M"]  # Change to "150M" or "650M" for better quality

# Embedding dimensions for each model (auto-detected, but listed here for reference)
ESM_EMBEDDING_DIMS = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
}

# ====================================================================
# PROCESSING CONFIGURATION
# ====================================================================

# Maximum sequence length (ESM2 can handle up to ~1024, but 512 is typical)
MAX_SEQUENCE_LENGTH = 512

# Batch size for processing multiple sequences
# Reduce if you get CUDA out-of-memory errors
BATCH_SIZE = 8

# Device configuration
def get_device():
    """Auto-detect GPU or use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_embedding_dim(model_name=None):
    """Get the embedding dimension for a specific ESM2 model."""
    model_name = model_name or ESM_MODEL_NAME
    return ESM_EMBEDDING_DIMS.get(model_name, 480)

# Print configuration on import
if __name__ == "__main__":
    print("ESM2 Configuration:")
    print(f"  Model: {ESM_MODEL_NAME}")
    print(f"  Embedding dim: {get_embedding_dim()}")
    print(f"  Device: {DEVICE}")
    print(f"  Max length: {MAX_SEQUENCE_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
