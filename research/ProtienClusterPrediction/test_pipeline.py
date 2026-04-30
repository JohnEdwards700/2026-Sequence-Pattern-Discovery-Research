"""
Exploratory clustering of WGS reads to identify resistance-associated patterns.

- Input: FASTA files (one per isolate)
- Method: k-mer encoding + autoencoder + clustering
- Output: cluster summaries and resistance motif signals

This pipeline is hypothesis-generating, not confirmatory.
"""

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from esm_embedder import ESMEmbedder
from Bio import SeqIO
import os
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

############################################
# CONFIGURATION
############################################

ISOLATE_FASTAS = {
    "isolate_main": "data/dataset.fasta",
    "background_1": "data/K22_sequence.fasta",
    "background_2": "data/K31_sequence.fasta",
}

K = 5
LATENT_DIM = 32
BATCH_SIZE = 512
EPOCHS = 5
N_CLUSTERS = 6
MAX_READS_PER_ISOLATE = 200000
RANDOM_STATE = 42

# Search these joint PCA/k-means settings and keep the best silhouette score.
PCA_COMPONENT_OPTIONS = [2, 4, 8, 16, 32, 64, 128]
KMEANS_CLUSTER_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12]
SILHOUETTE_SAMPLE_SIZE = 100000

def get_device():
    """
    Prefer CUDA when available, otherwise run on CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_device_cache(device):
    """
    Clear CUDA cache only when the active device is a CUDA device.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


DEVICE = get_device()
USE_CUDA = DEVICE.type == "cuda"

############################################
# ESM2 Embedding
############################################

def mean_pool(embeddings, masks):
    """
    Calculate the mean of embeddings while ignoring padding.
    """
    # Calculate the sum of embeddings multiplied by masks
    masked_embeddings = embeddings * masks.unsqueeze(-1)
    # Calculate the sum of masked embeddings
    sum_embeddings = masked_embeddings.sum(dim=1)
    # Calculate the sum of masks
    sum_masks = masks.sum(dim=1, keepdim=True)
    # Calculate the mean of embeddings
    mean_embeddings = sum_embeddings / sum_masks
    return mean_embeddings

def embed_dataset(file_path, embedder, cache_file=None, chunk_size=256, max_sequences=None):
    sequences = [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        data = torch.load(cache_file, map_location="cpu")
        return sequences, data["embeddings"]

    all_embeddings = []

    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i + chunk_size]

        emb, mask = embedder.embed_sequences(chunk)

        # pooled shape: (batch, 480)
        pooled = mean_pool(emb, mask).cpu()

        all_embeddings.append(pooled)

        # IMPORTANT: free memory immediately
        del emb
        del mask
        del pooled
        clear_device_cache(DEVICE)

        print(f"Processed {min(i + chunk_size, len(sequences))}/{len(sequences)}")

    embeddings = torch.cat(all_embeddings, dim=0)

    print("Final embedding shape:", embeddings.shape)
    # should be (N, 480)

    if cache_file:
        torch.save({"embeddings": embeddings}, cache_file)
        print(f"Saved embeddings to {cache_file}")

    return sequences, embeddings

class EmbeddingTransformer(nn.Module):
    def __init__(self, input_dim=512, model_dim=256, nhead=8, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x
    
def mask_embeddings(x, mask_ratio=0.15):
    mask = torch.rand_like(x) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

def create_sequences(embeddings, seq_len=32):
    chunks = []
    indices = []

    n_chunks = len(embeddings) // seq_len

    for i in range(n_chunks):
        start = i * seq_len
        end = start + seq_len

        chunks.append(embeddings[start:end])
        indices.append((start, end))

    return torch.stack(chunks), indices


def choose_best_pca_kmeans_train_val(
    X_train,
    X_val,
    pca_components_options,
    kmeans_cluster_options,
):
    best_result = None
    results = []

    print("\nSelecting PCA dimensions and k using validation silhouette score...")

    max_components = min(X_train.shape[0], X_train.shape[1])

    for n_components in pca_components_options:
        if n_components < 1 or n_components > max_components:
            print(f"  Skipping PCA={n_components}: valid range is 1-{max_components}")
            continue

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        explained_variance = float(np.sum(pca.explained_variance_ratio_))

        for n_clusters in kmeans_cluster_options:
            if n_clusters < 2 or n_clusters >= len(X_train_pca):
                print(f"  Skipping PCA={n_components}, k={n_clusters}: need 2 <= k < n_train")
                continue

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
            )

            train_labels = kmeans.fit_predict(X_train_pca)
            val_labels = kmeans.predict(X_val_pca)

            if len(np.unique(val_labels)) < 2:
                print(f"  Skipping PCA={n_components}, k={n_clusters}: only one cluster predicted on val")
                continue

            val_score = silhouette_score(X_val_pca, val_labels)

            result = {
                "pca_components": n_components,
                "n_clusters": n_clusters,
                "val_silhouette": float(val_score),
                "explained_variance": explained_variance,
                "pca": pca,
                "kmeans": kmeans,
                "train_labels": train_labels,
                "val_labels": val_labels,
            }
            results.append(result)

            print(
                f"  PCA={n_components:>3}, k={n_clusters:>2}, "
                f"val_silhouette={val_score:.4f}, explained_variance={explained_variance:.2%}"
            )

            if best_result is None or val_score > best_result["val_silhouette"]:
                best_result = result

    if best_result is None:
        raise ValueError("No valid PCA/k-means setting found.")

    with open("pca_kmeans_val_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pca_components", "n_clusters", "val_silhouette", "explained_variance"]
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "pca_components": result["pca_components"],
                "n_clusters": result["n_clusters"],
                "val_silhouette": result["val_silhouette"],
                "explained_variance": result["explained_variance"],
            })

    print(
        "\nBest validation setting: "
        f"PCA={best_result['pca_components']}, "
        f"k={best_result['n_clusters']}, "
        f"val_silhouette={best_result['val_silhouette']:.4f}, "
        f"explained_variance={best_result['explained_variance']:.2%}"
    )
    print("Saved validation search results to pca_kmeans_val_scores.csv")

    return best_result, results


print(f"Using device: {DEVICE}")

embedder = ESMEmbedder(device=DEVICE)

# Embed main dataset with caching
dataset1_file = ISOLATE_FASTAS["isolate_main"]
dataset1_cache = "cache/output_embeddings.pt"
dataset1_seqs, dataset1_emb = embed_dataset(
    dataset1_file,
    embedder,
    cache_file=dataset1_cache,
    max_sequences=MAX_READS_PER_ISOLATE
)

# Embed background datasets with caching
background_files = {
    "K22": ISOLATE_FASTAS["background_1"],
    "K31": ISOLATE_FASTAS["background_2"]
}
background_embs = {}
background_seqs = {}

for name, path in background_files.items():
    cache_file = f"cache/{name}_embeddings.pt"
    seqs, emb = embed_dataset(
        path,
        embedder,
        cache_file=cache_file,
        max_sequences=MAX_READS_PER_ISOLATE
    )
    background_seqs[name] = seqs
    background_embs[name] = emb

# Combine all isolates into one shared embedding space
all_embeddings = torch.cat(
    [dataset1_emb, background_embs["K22"], background_embs["K31"]],
    dim=0
)

all_seqs = (
    dataset1_seqs
    + background_seqs["K22"]
    + background_seqs["K31"]
)

all_isolate_labels = (
    ["isolate_main"] * len(dataset1_seqs)
    + ["background_1"] * len(background_seqs["K22"])
    + ["background_2"] * len(background_seqs["K31"])
)

# Pretrain the transformer model
print("Training transformer... 🚀")

seq_data, seq_indices = create_sequences(all_embeddings)

loader = DataLoader(
    seq_data,
    batch_size=32,
    shuffle=True,
    pin_memory=USE_CUDA,
)

transformer = EmbeddingTransformer(
    input_dim=dataset1_emb.shape[1]
).to(DEVICE)

optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

transformer.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)

        masked, mask = mask_embeddings(batch)

        output = transformer(masked)

        loss = loss_fn(output[mask], batch[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

print("Encoding embeddings with transformer... 🧠")

transformer.eval()

with torch.no_grad():
    seq_data, seq_indices = create_sequences(all_embeddings)

    seq_data = seq_data.to(DEVICE)

    encoded = transformer.input_proj(seq_data)
    encoded = transformer.transformer(encoded)

    # encoded shape: (num_chunks, seq_len, model_dim)
    encoded = encoded.cpu()

# Flatten back to one embedding per read
encoded_reads = []

for chunk in encoded:
    for read_embedding in chunk:
        encoded_reads.append(read_embedding.numpy())

encoded_emb = np.array(encoded_reads)

# Keep labels/sequences aligned with encoded embeddings
valid_n = len(encoded_emb)
encoded_emb = encoded_emb[:valid_n]
all_seqs = all_seqs[:valid_n]
all_isolate_labels = all_isolate_labels[:valid_n]

print("Encoded embedding shape:", encoded_emb.shape)
print("Number of labels:", len(all_isolate_labels))

X = encoded_emb
y_iso = np.array(all_isolate_labels)

X_train, X_temp, y_train_iso, y_temp_iso = train_test_split(
    X,
    y_iso,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y_iso
)

X_val, X_test, y_val_iso, y_test_iso = train_test_split(
    X_temp,
    y_temp_iso,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp_iso
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Select PCA dimension and k using validation silhouette score
best_clustering, clustering_search_results = choose_best_pca_kmeans_train_val(
    X_train,
    X_val,
    PCA_COMPONENT_OPTIONS,
    KMEANS_CLUSTER_OPTIONS,
)

best_pca = best_clustering["pca"]
best_kmeans = best_clustering["kmeans"]

# Final test performance
X_test_pca = best_pca.transform(X_test)
test_cluster_ids = best_kmeans.predict(X_test_pca)

if len(np.unique(test_cluster_ids)) < 2:
    print("Test set produced only one cluster; silhouette score is undefined.")
    test_silhouette = float("nan")
else:
    test_silhouette = silhouette_score(X_test_pca, test_cluster_ids)

print("\nFinal test performance:")
print(f"  Test silhouette score: {test_silhouette:.4f}")

# Also predict clusters for the full dataset for summaries / plotting
X_all_pca = best_pca.transform(encoded_emb)
cluster_ids = best_kmeans.predict(X_all_pca)
N_CLUSTERS = best_clustering["n_clusters"]

############################################
# CLUSTER SUMMARIES
############################################

cluster_summaries = defaultdict(lambda: {"sequences": [], "isolate_counts": Counter(), "gc_content": [], "motifs": Counter()})

def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

for i, (seq, iso_label) in enumerate(zip(all_seqs, all_isolate_labels)):
    if i >= len(cluster_ids):
        break

    cluster = cluster_ids[i]
    cluster_summaries[cluster]["sequences"].append(seq)
    cluster_summaries[cluster]["isolate_counts"][iso_label] += 1
    cluster_summaries[cluster]["gc_content"].append(gc_content(seq))

for cluster, summary in cluster_summaries.items():
    print(f"Cluster {cluster}:")
    print(f"  Sequences: {len(summary['sequences'])}")
    print(f"  Isolate counts: {summary['isolate_counts']}")
    print(f"  GC content: {np.mean(summary['gc_content']):.2f} ± {np.std(summary['gc_content']):.2f}")
    print(f"  Motifs: {summary['motifs']}")

############################################
# PCA VISUALIZATION
############################################

# 2D PCA visualization for the full dataset
plot_pca = PCA(n_components=2, random_state=RANDOM_STATE)
reduced_emb = plot_pca.fit_transform(encoded_emb)

plt.figure(figsize=(8, 6))
plt.scatter(
    reduced_emb[:, 0],
    reduced_emb[:, 1],
    c=cluster_ids,
    cmap="rainbow",
    alpha=0.5
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    f"2D PCA visualization | best val PCA={best_clustering['pca_components']}, "
    f"k={best_clustering['n_clusters']}, "
    f"test silhouette={test_silhouette:.4f}"
)
plt.colorbar(label="Cluster ID")
plt.show()