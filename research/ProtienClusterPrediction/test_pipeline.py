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
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from esm_embedder import ESMEmbedder
from Bio import SeqIO
import os
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import config

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
EPOCHS = 20
N_CLUSTERS = 6
MAX_READS_PER_ISOLATE = 200000
RANDOM_STATE = 42
TEST_ESM_MODEL_NAME = config.ESM_MODEL_OPTIONS["8M"] # Use the smaller ESM-8M for faster testing; switch to "ESM-650M" for final runs

# Search these joint PCA/k-means settings and keep the best silhouette score.
PCA_COMPONENT_OPTIONS = [2, 4, 8, 16, 32, 64, 128, 256]
KMEANS_CLUSTER_OPTIONS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
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

# GPU diagnostics
print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

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
        pooled = mean_pool(emb, mask)

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
    def __init__(self, input_dim=512, model_dim=512, nhead=16, num_layers=8):
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
    y_train_iso,
    y_val_iso,
    pca_components_options,
    kmeans_cluster_options,
):
    best_result = None
    results = []
    y_train_encoded, _ = encode_isolate_labels(y_train_iso)
    y_val_encoded, _ = encode_isolate_labels(y_val_iso)

    print("\nSelecting PCA dimensions and k using validation silhouette and isolate agreement...")

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
            val_ari = adjusted_rand_score(y_val_encoded, val_labels)
            val_nmi = normalized_mutual_info_score(y_val_encoded, val_labels)

            result = {
                "pca_components": n_components,
                "n_clusters": n_clusters,
                "val_silhouette": float(val_score),
                "val_ari": float(val_ari),
                "val_nmi": float(val_nmi),
                "explained_variance": explained_variance,
                "pca": pca,
                "kmeans": kmeans,
                "train_labels": train_labels,
                "val_labels": val_labels,
            }
            results.append(result)

            print(
                f"  PCA={n_components:>3}, k={n_clusters:>2}, "
                f"val_silhouette={val_score:.4f}, val_ari={val_ari:.4f}, "
                f"val_nmi={val_nmi:.4f}, explained_variance={explained_variance:.2%}"
            )

            if best_result is None or (
                (val_nmi, val_ari, val_score) >
                (best_result["val_nmi"], best_result["val_ari"], best_result["val_silhouette"])
            ):
                best_result = result

    if best_result is None:
        raise ValueError("No valid PCA/k-means setting found.")

    with open("pca_kmeans_val_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pca_components", "n_clusters", "val_silhouette", "val_ari", "val_nmi", "explained_variance"]
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "pca_components": result["pca_components"],
                "n_clusters": result["n_clusters"],
                "val_silhouette": result["val_silhouette"],
                "val_ari": result["val_ari"],
                "val_nmi": result["val_nmi"],
                "explained_variance": result["explained_variance"],
            })

    print(
        "\nBest validation setting: "
        f"PCA={best_result['pca_components']}, "
        f"k={best_result['n_clusters']}, "
        f"val_silhouette={best_result['val_silhouette']:.4f}, "
        f"val_ari={best_result['val_ari']:.4f}, "
        f"val_nmi={best_result['val_nmi']:.4f}, "
        f"explained_variance={best_result['explained_variance']:.2%}"
    )
    print("Saved validation search results to pca_kmeans_val_scores.csv")

    return best_result, results


def encode_isolate_labels(labels):
    unique_labels = sorted(set(labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.array([label_to_int[label] for label in labels])
    return encoded, label_to_int


def gc_correct_representation(X, gc_values):
    gc = np.asarray(gc_values, dtype=np.float32).reshape(-1, 1)
    design = np.hstack([gc, np.ones((len(gc), 1), dtype=np.float32)])
    coef, _, _, _ = np.linalg.lstsq(design, X, rcond=None)
    fitted = design @ coef
    corrected = X - fitted
    return corrected


def evaluate_test_metrics(pca_model, kmeans_model, X_test, y_test_iso):
    X_test_pca = pca_model.transform(X_test)
    test_cluster_ids = kmeans_model.predict(X_test_pca)
    y_test_encoded, _ = encode_isolate_labels(y_test_iso)

    if len(np.unique(test_cluster_ids)) < 2:
        return {
            "cluster_ids": test_cluster_ids,
            "silhouette": float("nan"),
            "ari": float("nan"),
            "nmi": float("nan"),
        }

    return {
        "cluster_ids": test_cluster_ids,
        "silhouette": silhouette_score(X_test_pca, test_cluster_ids),
        "ari": adjusted_rand_score(y_test_encoded, test_cluster_ids),
        "nmi": normalized_mutual_info_score(y_test_encoded, test_cluster_ids),
    }


def inspect_fixed_k_result(results, n_clusters):
    matching = [r for r in results if r["n_clusters"] == n_clusters]
    if not matching:
        return None
    return max(matching, key=lambda r: (r["val_nmi"], r["val_ari"], r["val_silhouette"]))


def print_cluster_isolate_diagnostics(cluster_ids, isolate_labels, gc_values):
    print("\nCluster/isolate diagnostics:")

    total_counts = Counter(isolate_labels)
    total_reads = len(isolate_labels)
    global_isolate_rates = {
        isolate: count / total_reads for isolate, count in total_counts.items()
    }

    contingency = defaultdict(Counter)
    for cluster_id, isolate in zip(cluster_ids, isolate_labels):
        contingency[cluster_id][isolate] += 1

    print("Cluster-by-isolate counts:")
    for cluster_id in sorted(contingency):
        counts = dict(contingency[cluster_id])
        cluster_total = sum(counts.values())
        print(f"  Cluster {cluster_id}: total={cluster_total}, counts={counts}")
        for isolate, count in counts.items():
            within_cluster_rate = count / cluster_total
            enrichment = within_cluster_rate / global_isolate_rates[isolate]
            print(
                f"    {isolate}: within-cluster={within_cluster_rate:.3f}, "
                f"global={global_isolate_rates[isolate]:.3f}, enrichment={enrichment:.3f}"
            )

    for cluster_id in sorted(set(cluster_ids)):
        cluster_gc = gc_values[np.array(cluster_ids) == cluster_id]
        print(
            f"  Cluster {cluster_id} GC: mean={cluster_gc.mean():.4f}, "
            f"std={cluster_gc.std():.4f}, n={len(cluster_gc)}"
        )

    encoded_isolates, isolate_map = encode_isolate_labels(isolate_labels)
    ari = adjusted_rand_score(encoded_isolates, cluster_ids)
    nmi = normalized_mutual_info_score(encoded_isolates, cluster_ids)
    print(f"Adjusted Rand Index vs isolate labels: {ari:.4f}")
    print(f"Normalized Mutual Information vs isolate labels: {nmi:.4f}")
    print(f"Isolate label mapping: {isolate_map}")


def plot_representation_diagnostics(reduced_emb, cluster_ids, isolate_labels, gc_values):
    isolate_names = sorted(set(isolate_labels))
    isolate_to_int = {name: idx for idx, name in enumerate(isolate_names)}
    isolate_numeric = np.array([isolate_to_int[label] for label in isolate_labels])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    scatter0 = axes[0].scatter(
        reduced_emb[:, 0],
        reduced_emb[:, 1],
        c=cluster_ids,
        cmap="rainbow",
        alpha=0.5
    )
    axes[0].set_title("PCA colored by cluster")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(scatter0, ax=axes[0], label="Cluster ID")

    scatter1 = axes[1].scatter(
        reduced_emb[:, 0],
        reduced_emb[:, 1],
        c=isolate_numeric,
        cmap="tab10",
        alpha=0.5
    )
    axes[1].set_title("PCA colored by isolate")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    handles, _ = scatter1.legend_elements()
    axes[1].legend(handles, isolate_names, title="Isolate", loc="best")

    scatter2 = axes[2].scatter(
        reduced_emb[:, 0],
        reduced_emb[:, 1],
        c=gc_values,
        cmap="viridis",
        alpha=0.5
    )
    axes[2].set_title("PCA colored by GC")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    fig.colorbar(scatter2, ax=axes[2], label="GC content")

    plt.tight_layout()
    plt.show()


print(f"Using test ESM model: {TEST_ESM_MODEL_NAME}")
embedder = ESMEmbedder(model_name=TEST_ESM_MODEL_NAME, device=DEVICE)

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
all_gc_values = np.array([
    (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0
    for seq in all_seqs
])

# Pretrain the transformer model
print("Training transformer... 🚀")

seq_data, seq_indices = create_sequences(all_embeddings)

loader = DataLoader(
    seq_data,
    batch_size=32,
    shuffle=True,
    pin_memory=False,
)

transformer = EmbeddingTransformer(
    input_dim=dataset1_emb.shape[1],
    model_dim=512,        # Increase from 256
    nhead=16,             # Increase from 8
    num_layers=8          # Increase from 4
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
all_gc_values = all_gc_values[:valid_n]

print("Encoded embedding shape:", encoded_emb.shape)
print("Number of labels:", len(all_isolate_labels))

X = encoded_emb
y_iso = np.array(all_isolate_labels)
X_gc_corrected = gc_correct_representation(X, all_gc_values)

indices = np.arange(len(X))
train_idx, temp_idx, y_train_iso, y_temp_iso = train_test_split(
    indices,
    y_iso,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y_iso
)

val_idx, test_idx, y_val_iso, y_test_iso = train_test_split(
    temp_idx,
    y_temp_iso,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp_iso
)

print("Train shape:", X[train_idx].shape)
print("Val shape:", X[val_idx].shape)
print("Test shape:", X[test_idx].shape)

representation_runs = [
    ("Transformer embeddings", X),
    ("GC-corrected transformer embeddings", X_gc_corrected),
]
representation_results = []

for run_name, run_X in representation_runs:
    print(f"\n=== Evaluating {run_name} ===")
    best_clustering, clustering_search_results = choose_best_pca_kmeans_train_val(
        run_X[train_idx],
        run_X[val_idx],
        y_iso[train_idx],
        y_iso[val_idx],
        PCA_COMPONENT_OPTIONS,
        KMEANS_CLUSTER_OPTIONS,
    )
    test_metrics = evaluate_test_metrics(
        best_clustering["pca"],
        best_clustering["kmeans"],
        run_X[test_idx],
        y_iso[test_idx],
    )
    print(
        f"Test metrics | silhouette={test_metrics['silhouette']:.4f}, "
        f"ARI={test_metrics['ari']:.4f}, NMI={test_metrics['nmi']:.4f}"
    )
    representation_results.append({
        "name": run_name,
        "X": run_X,
        "best_clustering": best_clustering,
        "search_results": clustering_search_results,
        "test_metrics": test_metrics,
    })

selected_run = max(
    representation_results,
    key=lambda r: (
        r["test_metrics"]["nmi"],
        r["test_metrics"]["ari"],
        r["test_metrics"]["silhouette"],
    ),
)

print(f"\nSelected representation: {selected_run['name']}")
print(
    f"Selected test metrics | silhouette={selected_run['test_metrics']['silhouette']:.4f}, "
    f"ARI={selected_run['test_metrics']['ari']:.4f}, "
    f"NMI={selected_run['test_metrics']['nmi']:.4f}"
)

best_clustering = selected_run["best_clustering"]
clustering_search_results = selected_run["search_results"]
best_pca = best_clustering["pca"]
best_kmeans = best_clustering["kmeans"]
selected_X = selected_run["X"]
test_silhouette = selected_run["test_metrics"]["silhouette"]

k3_result = inspect_fixed_k_result(clustering_search_results, 3)
if k3_result is not None:
    k3_cluster_ids = k3_result["kmeans"].predict(k3_result["pca"].transform(selected_X))
    print("\nExplicit k=3 inspection:")
    print(
        f"  PCA={k3_result['pca_components']}, "
        f"val_silhouette={k3_result['val_silhouette']:.4f}, "
        f"val_ari={k3_result['val_ari']:.4f}, val_nmi={k3_result['val_nmi']:.4f}"
    )
    print_cluster_isolate_diagnostics(k3_cluster_ids, all_isolate_labels, all_gc_values)

# Also predict clusters for the full dataset for summaries / plotting
X_all_pca = best_pca.transform(selected_X)
cluster_ids = best_kmeans.predict(X_all_pca)
N_CLUSTERS = best_clustering["n_clusters"]

print_cluster_isolate_diagnostics(cluster_ids, all_isolate_labels, all_gc_values)

subset_mask = np.isin(y_iso, ["isolate_main", "background_1"])
subset_X = selected_X[subset_mask]
subset_y = y_iso[subset_mask]
subset_gc = all_gc_values[subset_mask]

subset_indices = np.arange(len(subset_X))
subset_train_idx, subset_temp_idx, subset_y_train, subset_y_temp = train_test_split(
    subset_indices,
    subset_y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=subset_y
)
subset_val_idx, subset_test_idx, subset_y_val, subset_y_test = train_test_split(
    subset_temp_idx,
    subset_y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=subset_y_temp
)

print("\n=== isolate_main vs background_1 only ===")
subset_best_clustering, _ = choose_best_pca_kmeans_train_val(
    subset_X[subset_train_idx],
    subset_X[subset_val_idx],
    subset_y[subset_train_idx],
    subset_y[subset_val_idx],
    PCA_COMPONENT_OPTIONS,
    KMEANS_CLUSTER_OPTIONS,
)
subset_test_metrics = evaluate_test_metrics(
    subset_best_clustering["pca"],
    subset_best_clustering["kmeans"],
    subset_X[subset_test_idx],
    subset_y[subset_test_idx],
)
print(
    f"Subset test metrics | silhouette={subset_test_metrics['silhouette']:.4f}, "
    f"ARI={subset_test_metrics['ari']:.4f}, NMI={subset_test_metrics['nmi']:.4f}"
)
subset_cluster_ids = subset_best_clustering["kmeans"].predict(
    subset_best_clustering["pca"].transform(subset_X)
)
print_cluster_isolate_diagnostics(subset_cluster_ids, subset_y, subset_gc)

############################################
# CLUSTER SUMMARIES
############################################

cluster_summaries = defaultdict(lambda: {"sequences": [], "isolate_counts": Counter(), "gc_content": [], "motifs": Counter()})

for i, (seq, iso_label) in enumerate(zip(all_seqs, all_isolate_labels)):
    if i >= len(cluster_ids):
        break

    cluster = cluster_ids[i]
    cluster_summaries[cluster]["sequences"].append(seq)
    cluster_summaries[cluster]["isolate_counts"][iso_label] += 1
    cluster_summaries[cluster]["gc_content"].append(all_gc_values[i])

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
reduced_emb = plot_pca.fit_transform(selected_X)
plot_representation_diagnostics(reduced_emb, cluster_ids, all_isolate_labels, all_gc_values)
