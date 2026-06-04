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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from viz_tsne import plot_tsne

############################################
# CONFIGURATION
############################################

ISOLATE_FASTAS = {
    "isolate_main": "data/dataset.fasta",
    "background_1": "data/K22_sequence.fasta",
    "background_2": "data/K31_sequence.fasta",
}

K = 5  # k-mer size
LATENT_DIM = 16 # Dimension of autoencoder latent space
BATCH_SIZE = 512 # Adjust based on memory constraints
EPOCHS = 5 
N_CLUSTERS = 6
CLUSTER_OPTIONS = range(2, 13)
REPRESENTATION = "autoencoder"  # "autoencoder" or "pca" are the options for clustering features
PCA_COMPONENTS = 16
MAX_READS_PER_ISOLATE = 200_000
RANDOM_STATE = 42
VAL_SIZE = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# k-mer encoding
############################################

BASES = ["A", "C", "G", "T"]

def generate_kmer_index(k):
    kmers = ["".join(p) for p in __product(BASES, k)]
    return {kmer: i for i, kmer in enumerate(kmers)}

def __product(chars, k):
    if k == 1:
        return chars
    return [c + p for c in chars for p in __product(chars, k - 1)]

KMER_INDEX = generate_kmer_index(K)
INPUT_DIM = len(KMER_INDEX)

def encode_kmers(seq):
    vec = np.zeros(INPUT_DIM, dtype=np.float32)
    seq = seq.upper()
    for i in range(len(seq) - K + 1):
        kmer = seq[i:i+K]
        if "N" in kmer:
            continue
        idx = KMER_INDEX.get(kmer)
        if idx is not None:
            vec[idx] += 1
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec

def gc_content(seq):
    seq = seq.upper()
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)

############################################
# FASTA streaming dataset
############################################

class FastaDataset(IterableDataset):
    def __init__(self, fasta_path, isolate_id, max_reads):
        self.fasta_path = fasta_path
        self.isolate_id = isolate_id
        self.max_reads = max_reads

    def __iter__(self):
        with open(self.fasta_path) as f:
            seq = ""
            count = 0
            for line in f:
                if line.startswith(">"):
                    if seq:
                        yield {
                            "kmer": encode_kmers(seq),
                            "gc": gc_content(seq),
                            "isolate": self.isolate_id
                        }
                        count += 1
                        if count >= self.max_reads:
                            return
                    seq = ""
                else:
                    seq += line.strip()
            if seq and count < self.max_reads:
                yield {
                    "kmer": encode_kmers(seq),
                    "gc": gc_content(seq),
                    "isolate": self.isolate_id
                }

############################################
# Autoencoder model
############################################

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

############################################
# LOAD DATA
############################################

print("Loading data...")

all_records = []
for isolate, fasta in ISOLATE_FASTAS.items():
    ds = FastaDataset(fasta, isolate, MAX_READS_PER_ISOLATE)
    for record in ds:
        all_records.append(record)

kmer_matrix = np.vstack([r["kmer"] for r in all_records])
gc_values = np.array([r["gc"] for r in all_records])
isolate_labels = [r["isolate"] for r in all_records]

############################################
# TRAIN AUTOENCODER
############################################

print("Training autoencoder...")

model = AutoEncoder(INPUT_DIM, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

dataset_tensor = torch.tensor(kmer_matrix, dtype=torch.float32)
loader = DataLoader(dataset_tensor, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    batch_losses = []
    for batch in loader:
        batch = batch.to(DEVICE)
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_mean_loss = float(np.mean(batch_losses))
    epoch_min_loss = float(np.min(batch_losses))
    epoch_max_loss = float(np.max(batch_losses))
    print(
        f"Epoch {epoch+1}/{EPOCHS} - "
        f"mean loss: {epoch_mean_loss:.8f} | "
        f"min batch loss: {epoch_min_loss:.8f} | "
        f"max batch loss: {epoch_max_loss:.8f}"
    )

############################################
# EMBEDDING + CLUSTERING
############################################

print("Embedding and clustering...")

model.eval()
with torch.no_grad():
    dataset_on_device = dataset_tensor.to(DEVICE)
    reconstructions, latent_tensor = model(dataset_on_device)
    embeddings = latent_tensor.cpu().numpy()

recon_np = reconstructions.cpu().numpy()
sample_mse = np.mean((recon_np - kmer_matrix) ** 2, axis=1)
latent_var = np.var(embeddings, axis=0)
latent_norms = np.linalg.norm(embeddings, axis=1)
collapsed_dims = int(np.sum(latent_var < 1e-8))

print("Autoencoder diagnostics:")
print(
    f"  Reconstruction MSE per read - mean: {sample_mse.mean():.8f}, "
    f"median: {np.median(sample_mse):.8f}, std: {sample_mse.std():.8f}, "
    f"min: {sample_mse.min():.8f}, max: {sample_mse.max():.8f}"
)
print(
    f"  Latent variance per dimension - mean: {latent_var.mean():.8f}, "
    f"min: {latent_var.min():.8f}, max: {latent_var.max():.8f}, "
    f"near-zero dims (<1e-8): {collapsed_dims}/{LATENT_DIM}"
)
print(
    f"  Latent vector L2 norms - mean: {latent_norms.mean():.8f}, "
    f"std: {latent_norms.std():.8f}, min: {latent_norms.min():.8f}, "
    f"max: {latent_norms.max():.8f}"
)

if REPRESENTATION == "autoencoder":
    clustering_features = embeddings
    representation_name = "Autoencoder"
elif REPRESENTATION == "pca":
    n_components = min(PCA_COMPONENTS, kmer_matrix.shape[0], kmer_matrix.shape[1])
    pca_model = PCA(n_components=n_components, random_state=RANDOM_STATE)
    clustering_features = pca_model.fit_transform(kmer_matrix)
    representation_name = f"PCA ({n_components} components)"
    explained_variance = float(np.sum(pca_model.explained_variance_ratio_))
    print("PCA diagnostics:")
    print(f"  Components used: {n_components}")
    print(f"  Total explained variance: {explained_variance:.4f}")
else:
    raise ValueError(f"Unknown REPRESENTATION: {REPRESENTATION}")

print(f"Using representation for clustering: {representation_name}")

# t-SNE visualization on the selected representation
plot_tsne(
    clustering_features,
    isolate_labels,
    title=f"t-SNE on {representation_name} features",
    perplexity=30,
)

X_train, X_val, y_train, y_val = train_test_split(
    clustering_features,
    isolate_labels,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=isolate_labels,
)

best_kmeans = None
best_val_silhouette = float("-inf")
best_n_clusters = None

print("Searching cluster counts on validation split...")

for n_clusters in CLUSTER_OPTIONS:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(X_train)

    val_cluster_ids = kmeans.predict(X_val)
    if len(np.unique(val_cluster_ids)) < 2:
        print(f"  k={n_clusters}: skipped (only one cluster predicted on validation set)")
        continue

    val_silhouette = silhouette_score(X_val, val_cluster_ids)
    print(f"  k={n_clusters}: validation silhouette score = {val_silhouette:.4f}")

    if val_silhouette > best_val_silhouette:
        best_val_silhouette = val_silhouette
        best_n_clusters = n_clusters
        best_kmeans = kmeans

if best_kmeans is None:
    raise ValueError("No valid cluster count produced at least two validation clusters.")

N_CLUSTERS = best_n_clusters
print(f"Best validation silhouette score: {best_val_silhouette:.4f} at k={N_CLUSTERS}")

# Refit on the full embedding set for final cluster summaries.
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
cluster_ids = kmeans.fit_predict(clustering_features)

############################################
# CLUSTER SUMMARIES
############################################

cluster_summary = defaultdict(lambda: {
    "count": 0,
    "gc": [],
    "isolates": Counter(),
    "kmer_vectors": []
})

for cid, iso, gc, vec in zip(cluster_ids, isolate_labels, gc_values, kmer_matrix):
    cs = cluster_summary[cid]
    cs["count"] += 1
    cs["gc"].append(gc)
    cs["isolates"][iso] += 1
    cs["kmer_vectors"].append(vec)

############################################
# RESISTANCE MOTIF HEURISTICS
############################################

KNOWN_RESISTANCE_MOTIFS = {
    "blaKPC": ["TGGCG", "CGTGG"],
    "blaNDM": ["GGGCG", "GATCG"],
    "blaOXA": ["ACGAA", "TCGAC"],
    "blaVIM": ["GCGCG", "CGCGG"]
}

print("\n=== CLUSTER REPORT ===\n")

global_kmer_mean = np.mean(kmer_matrix, axis=0)

# build motif -> kmer index map (skip motifs not in index)
motif_idx_map = {gene: [KMER_INDEX[m] for m in motifs if m in KMER_INDEX] for gene, motifs in KNOWN_RESISTANCE_MOTIFS.items()}

# count motif hits per cluster per isolate (presence if k-mer count > 0)
from collections import defaultdict
motif_counts = defaultdict(lambda: defaultdict(Counter))  # motif_counts[cluster_id][isolate][gene] = count

for cid, iso, vec in zip(cluster_ids, isolate_labels, kmer_matrix):
    for gene, idxs in motif_idx_map.items():
        for idx in idxs:
            if vec[idx] > 0:
                motif_counts[cid][iso][gene] += 1
                break

for cid in sorted(cluster_summary.keys()):
    info = cluster_summary[cid]
    mean_gc = np.mean(info["gc"])
    mean_kmer = np.mean(info["kmer_vectors"], axis=0)
    enriched = mean_kmer - global_kmer_mean
    top_kmers = np.argsort(enriched)[-25:]

    # genes enriched in top kmers (cluster-level heuristic)
    hits = set()
    for gene, motifs in KNOWN_RESISTANCE_MOTIFS.items():
        for m in motifs:
            idx = KMER_INDEX.get(m)
            if idx is not None and idx in top_kmers:
                hits.add(gene)

    # per-isolate motif presence summary (counts of reads with motif k-mer)
    per_isolate_hits = {iso: dict(cnts) for iso, cnts in motif_counts[cid].items()}

    print(f"Cluster {cid} - Reads: {info['count']}")
    print(f"  Mean GC: {mean_gc:.3f}")
    print(f"  Isolate distribution: {dict(info['isolates'])}")
    print(f"  Resistance signals (cluster-level k-mer enrichment): {hits if hits else 'None detected'}")
    print(f"  Resistance signals (per-isolate read counts): {per_isolate_hits if per_isolate_hits else 'None detected'}")
    print("")

print("Analysis complete.")
