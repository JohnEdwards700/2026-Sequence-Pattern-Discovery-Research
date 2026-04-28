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
EPOCHS = 1 # 5
N_CLUSTERS = 6
MAX_READS_PER_ISOLATE = 200 # 200,000
RANDOM_STATE = 42

# Search these joint PCA/k-means settings and keep the best silhouette score.
PCA_COMPONENT_OPTIONS = [2, 8, 16] # 2, 4, 8, 16, 32, 64, 128
KMEANS_CLUSTER_OPTIONS = [2, 3, 4] # 2, 3, 4, 5, 6, 8, 10, 12
SILHOUETTE_SAMPLE_SIZE = 1000 # 100,000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        torch.cuda.empty_cache()

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


def choose_best_pca_kmeans(
    embeddings,
    pca_components_options,
    kmeans_cluster_options,
    silhouette_sample_size=None,
):
    """
    Jointly tune PCA dimensions and k-means clusters by maximizing silhouette.
    """
    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features)
    results = []
    best_result = None

    print("\nSelecting PCA dimensions and k by silhouette score...")

    for n_components in pca_components_options:
        if n_components < 1 or n_components > max_components:
            print(f"  Skipping PCA={n_components}: valid range is 1-{max_components}")
            continue

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        reduced_embeddings = pca.fit_transform(embeddings)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))

        for n_clusters in kmeans_cluster_options:
            if n_clusters < 2 or n_clusters >= n_samples:
                print(f"  Skipping PCA={n_components}, k={n_clusters}: silhouette needs 2 <= k < n_samples")
                continue

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_STATE,
            )
            labels = kmeans.fit_predict(reduced_embeddings)

            if len(np.unique(labels)) < 2:
                print(f"  Skipping PCA={n_components}, k={n_clusters}: only one cluster was produced")
                continue

            sample_size = silhouette_sample_size
            if sample_size is not None:
                sample_size = min(sample_size, n_samples)

            score = silhouette_score(
                reduced_embeddings,
                labels,
                sample_size=sample_size,
                random_state=RANDOM_STATE,
            )

            result = {
                "pca_components": n_components,
                "n_clusters": n_clusters,
                "silhouette_score": float(score),
                "explained_variance": explained_variance,
                "labels": labels,
                "embeddings": reduced_embeddings,
                "kmeans": kmeans,
                "pca": pca,
            }
            results.append(result)

            print(
                f"  PCA={n_components:>3}, k={n_clusters:>2}, "
                f"silhouette={score:.4f}, explained_variance={explained_variance:.2%}"
            )

            if best_result is None or score > best_result["silhouette_score"]:
                best_result = result

    if best_result is None:
        raise ValueError("No valid PCA/k-means setting produced a silhouette score.")

    with open("pca_kmeans_silhouette_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pca_components",
                "n_clusters",
                "silhouette_score",
                "explained_variance",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "pca_components": result["pca_components"],
                "n_clusters": result["n_clusters"],
                "silhouette_score": result["silhouette_score"],
                "explained_variance": result["explained_variance"],
            })

    print(
        "\nBest clustering setting: "
        f"PCA={best_result['pca_components']}, "
        f"k={best_result['n_clusters']}, "
        f"silhouette={best_result['silhouette_score']:.4f}, "
        f"explained_variance={best_result['explained_variance']:.2%}"
    )
    print("Saved silhouette search results to pca_kmeans_silhouette_scores.csv")

    return best_result, results


embedder = ESMEmbedder()

# Embed main dataset with caching
dataset1_file = ISOLATE_FASTAS["isolate_main"]
dataset1_cache = "cache/debug_output_embeddings.pt"
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
    cache_file = f"cache/debug_{name}_embeddings.pt"
    seqs, emb = embed_dataset(
        path,
        embedder,
        cache_file=cache_file,
        max_sequences=MAX_READS_PER_ISOLATE
    )
    background_seqs[name] = seqs
    background_embs[name] = emb

# Pretrain the transformer model
print("Training transformer... 🚀")

seq_data, seq_indices = create_sequences(dataset1_emb)

loader = DataLoader(seq_data, batch_size=32, shuffle=True)

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
    seq_data, seq_indices = create_sequences(dataset1_emb)

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

# Make sure we only keep as many reads as we originally had
encoded_emb = encoded_emb[:len(dataset1_seqs)]

print("Encoded embedding shape:", encoded_emb.shape)

# Perform PCA + clustering on the encoded embeddings, choosing the joint setting
# with the highest silhouette score.
best_clustering, clustering_search_results = choose_best_pca_kmeans(
    encoded_emb,
    PCA_COMPONENT_OPTIONS,
    KMEANS_CLUSTER_OPTIONS,
    silhouette_sample_size=SILHOUETTE_SAMPLE_SIZE,
)

pca = best_clustering["pca"]
kmeans = best_clustering["kmeans"]
cluster_ids = best_clustering["labels"]
cluster_embeddings = best_clustering["embeddings"]
N_CLUSTERS = best_clustering["n_clusters"]

############################################
# CLUSTER SUMMARIES
############################################

cluster_summaries = defaultdict(lambda: {"sequences": [], "isolate_counts": Counter(), "gc_content": [], "motifs": Counter()})

def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)

for i, seq in enumerate(dataset1_seqs[:len(cluster_ids)]):
    cluster = cluster_ids[i]
    cluster_summaries[cluster]["sequences"].append(seq)
    cluster_summaries[cluster]["isolate_counts"]["main"] += 1
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

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
reduced_emb = pca.fit_transform(encoded_emb)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    reduced_emb[:, 0],
    reduced_emb[:, 1],
    c=cluster_ids,
    cmap="rainbow",
    alpha=0.5,
    label="Clusters"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    f"2D PCA visualization | Best Search: PCA={best_clustering['pca_components']}, "
    f"k={best_clustering['n_clusters']}, "
    f"silhouette={best_clustering['silhouette_score']:.4f}"
)
plt.colorbar()
plt.show()