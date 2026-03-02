"""
Carbapenem Resistance Analysis Pipeline (Single File)
=====================================================
Analyzes WGS data from bacterial isolates to identify carbapenem resistance patterns.

Research Questions:
- Which resistance genes (blaKPC, blaNDM, blaVIM, blaOXA) are present?
- Are resistance genes chromosomal or plasmid-borne?
- Are there mutations in porins/efflux pumps/PBPs?
- Do isolates share identical or distinct resistance determinants?
- Are novel β-lactamase variants present?

Usage:
    python carbapenem_resistance_analysis.py
    python carbapenem_resistance_analysis.py --fastas iso1=path1.fasta iso2=path2.fasta
"""

import os
import sys
import csv
import argparse
import itertools
from collections import defaultdict, Counter
import numpy as np

# Optional: sklearn for clustering (fallback to simple methods if unavailable)
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. Install via: pip install scikit-learn")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input FASTA files - map isolate names to file paths
ISOLATE_FASTAS = {
    "isolate_main": "data/dataset.fasta",
    "background_1": "data/K22_sequence.fasta",
    "background_2": "data/K31_sequence.fasta",
}

# Analysis parameters
K = 5                          # k-mer length (5-mers capture local sequence patterns)
LATENT_DIM = 32                # dimensions for PCA reduction
N_CLUSTERS = 6                 # number of clusters for k-means
MAX_READS_PER_ISOLATE = 200000 # limit reads per file (memory management)
RANDOM_STATE = 42              # reproducibility seed

# Known carbapenem resistance gene motifs (short k-mer signatures)
# These are simplified markers - real analysis would use full gene sequences
KNOWN_RESISTANCE_MOTIFS = {
    "blaKPC": ["TGGCG", "CGTGG"],  # KPC carbapenemase markers
    "blaNDM": ["GGGCG", "GATCG"],  # NDM metallo-β-lactamase markers
    "blaOXA": ["ACGAA", "TCGAC"],  # OXA-48-like markers
    "blaVIM": ["GCGCG", "CGCGG"],  # VIM metallo-β-lactamase markers
}

# =============================================================================
# K-MER ENCODING FUNCTIONS
# =============================================================================

def generate_kmer_index(k, bases=("A", "C", "G", "T")):
    """
    Create a dictionary mapping all possible k-mers to unique indices.
    
    For k=5, this creates 4^5 = 1024 possible k-mers.
    Example: {"AAAAA": 0, "AAAAC": 1, ..., "TTTTT": 1023}
    """
    kmers = [''.join(combo) for combo in itertools.product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(kmers)}

# Pre-generate k-mer index for the configured K value
KMER_INDEX = generate_kmer_index(K)
INPUT_DIM = len(KMER_INDEX)  # 1024 for k=5

def encode_kmers(sequence):
    """
    Convert a DNA sequence into a normalized k-mer frequency vector.
    
    Args:
        sequence: DNA string (e.g., "ATCGATCG")
    
    Returns:
        numpy array of shape (INPUT_DIM,) with normalized k-mer counts
    """
    vector = np.zeros(INPUT_DIM, dtype=np.float32)
    seq = sequence.upper()
    
    # Slide window across sequence, count each k-mer
    for i in range(len(seq) - K + 1):
        kmer = seq[i:i + K]
        
        # Skip k-mers containing N (unknown base)
        if "N" in kmer:
            continue
        
        idx = KMER_INDEX.get(kmer)
        if idx is not None:
            vector[idx] += 1
    
    # Normalize to frequencies (sum to 1)
    total = vector.sum()
    if total > 0:
        vector /= total
    
    return vector

def calculate_gc_content(sequence):
    """
    Calculate GC content (proportion of G and C bases).
    
    GC content varies between species and can indicate:
    - Horizontal gene transfer (if region differs from genome average)
    - Plasmid vs chromosomal origin
    """
    seq = sequence.upper()
    if not seq:
        return 0.0
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)

# =============================================================================
# FASTA FILE PARSING
# =============================================================================

def parse_fasta(filepath, max_reads=MAX_READS_PER_ISOLATE):
    """
    Stream reads from a FASTA file.
    
    FASTA format:
        >header_line
        SEQUENCE...
        >next_header
        SEQUENCE...
    
    Yields one sequence at a time for memory efficiency.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return
    
    with open(filepath, "r") as fh:
        current_sequence = ""
        read_count = 0
        
        for line in fh:
            line = line.strip()
            
            if line.startswith(">"):
                # Header line - yield previous sequence if exists
                if current_sequence:
                    yield current_sequence
                    read_count += 1
                    if read_count >= max_reads:
                        return
                current_sequence = ""
            else:
                # Sequence line - append to current sequence
                current_sequence += line
        
        # Don't forget the last sequence
        if current_sequence and read_count < max_reads:
            yield current_sequence

# =============================================================================
# CLUSTERING ANALYSIS
# =============================================================================

def perform_clustering(kmer_matrix):
    """
    Reduce dimensionality and cluster sequences.
    
    Steps:
    1. PCA: Reduce 1024 dimensions to LATENT_DIM (32)
    2. K-means: Group similar sequences into N_CLUSTERS
    
    Returns:
        embeddings: Low-dimensional representation
        cluster_ids: Cluster assignment for each sequence
    """
    if not HAS_SKLEARN:
        print("Error: scikit-learn required for clustering")
        return None, None
    
    # PCA reduces noise and speeds up clustering
    n_components = min(LATENT_DIM, kmer_matrix.shape[1], kmer_matrix.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embeddings = pca.fit_transform(kmer_matrix)
    
    print(f"  PCA explained variance: {sum(pca.explained_variance_ratio_):.2%}")
    
    # K-means groups sequences with similar k-mer profiles
    n_clusters = min(N_CLUSTERS, len(kmer_matrix))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    cluster_ids = kmeans.fit_predict(embeddings)
    
    return embeddings, cluster_ids

# =============================================================================
# RESISTANCE GENE DETECTION
# =============================================================================

def detect_resistance_motifs(kmer_vectors, cluster_ids, isolate_labels):
    """
    Scan for known resistance gene motifs in k-mer vectors.
    
    Two detection methods:
    1. Cluster-level: Find motifs enriched in cluster's top k-mers
    2. Per-read: Count reads containing each motif
    
    Returns:
        motif_counts: Dict of {cluster: {isolate: {gene: count}}}
    """
    # Map gene names to their motif indices in our k-mer vector
    motif_idx_map = {}
    for gene, motifs in KNOWN_RESISTANCE_MOTIFS.items():
        indices = [KMER_INDEX[m] for m in motifs if m in KMER_INDEX]
        motif_idx_map[gene] = indices
    
    # Count reads with each motif per cluster per isolate
    motif_counts = defaultdict(lambda: defaultdict(Counter))
    
    for cluster_id, isolate, kmer_vec in zip(cluster_ids, isolate_labels, kmer_vectors):
        for gene, indices in motif_idx_map.items():
            # Check if any motif k-mer is present in this read
            for idx in indices:
                if kmer_vec[idx] > 0:
                    motif_counts[cluster_id][isolate][gene] += 1
                    break  # Count read once per gene
    
    return motif_counts

# =============================================================================
# REPORTING
# =============================================================================

def generate_cluster_report(cluster_summary, motif_counts, global_kmer_mean):
    """
    Print and save cluster analysis results.
    
    For each cluster, reports:
    - Number of reads
    - Mean GC content
    - Distribution across isolates
    - Detected resistance signals
    """
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS REPORT")
    print("=" * 60 + "\n")
    
    summary_rows = []
    motif_rows = []
    
    for cluster_id in sorted(cluster_summary.keys()):
        info = cluster_summary[cluster_id]
        
        # Calculate cluster statistics
        mean_gc = np.mean(info["gc"]) if info["gc"] else 0.0
        mean_kmer = np.mean(info["kmer_vectors"], axis=0)
        
        # Find k-mers enriched in this cluster vs global average
        enrichment = mean_kmer - global_kmer_mean
        top_kmer_indices = set(np.argsort(enrichment)[-25:])  # Top 25 enriched
        
        # Check if resistance motifs are among enriched k-mers
        cluster_level_hits = set()
        for gene, motifs in KNOWN_RESISTANCE_MOTIFS.items():
            for motif in motifs:
                idx = KMER_INDEX.get(motif)
                if idx is not None and idx in top_kmer_indices:
                    cluster_level_hits.add(gene)
        
        # Get per-isolate motif counts
        per_isolate_hits = {
            iso: dict(counts) 
            for iso, counts in motif_counts[cluster_id].items()
        }
        
        # Print cluster summary
        print(f"CLUSTER {cluster_id}")
        print(f"  Reads: {info['count']:,}")
        print(f"  Mean GC: {mean_gc:.3f}")
        print(f"  Isolate distribution: {dict(info['isolates'])}")
        print(f"  Resistance signals (enriched k-mers): {cluster_level_hits or 'None detected'}")
        print(f"  Resistance signals (per-isolate): {per_isolate_hits or 'None detected'}")
        print()
        
        # Collect data for CSV export
        summary_rows.append({
            "cluster": cluster_id,
            "reads": info["count"],
            "mean_gc": f"{mean_gc:.3f}",
            "isolate_distribution": str(dict(info["isolates"])),
            "resistance_genes": ";".join(sorted(cluster_level_hits)) or "None"
        })
        
        for isolate, gene_counts in motif_counts[cluster_id].items():
            for gene, count in gene_counts.items():
                motif_rows.append({
                    "cluster": cluster_id,
                    "isolate": isolate,
                    "gene": gene,
                    "read_count": count
                })
    
    # Write CSV files
    with open("cluster_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cluster", "reads", "mean_gc", "isolate_distribution", "resistance_genes"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    
    with open("resistance_motifs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cluster", "isolate", "gene", "read_count"
        ])
        writer.writeheader()
        writer.writerows(motif_rows)
    
    print("Output files: cluster_summary.csv, resistance_motifs.csv")

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_analysis(fasta_files):
    """
    Main analysis pipeline.
    
    Steps:
    1. Load and encode sequences from all FASTA files
    2. Perform dimensionality reduction (PCA)
    3. Cluster sequences (K-means)
    4. Detect resistance motifs
    5. Generate report
    """
    print("\n" + "=" * 60)
    print("CARBAPENEM RESISTANCE ANALYSIS")
    print("=" * 60)
    
    # --- Step 1: Load and encode sequences ---
    print("\n[1/4] Loading and encoding sequences...")
    
    kmer_vectors = []
    gc_values = []
    isolate_labels = []
    
    for isolate_name, filepath in fasta_files.items():
        if not os.path.exists(filepath):
            print(f"  Skipping {isolate_name}: file not found ({filepath})")
            continue
        
        count = 0
        for sequence in parse_fasta(filepath):
            kmer_vectors.append(encode_kmers(sequence))
            gc_values.append(calculate_gc_content(sequence))
            isolate_labels.append(isolate_name)
            count += 1
        
        print(f"  {isolate_name}: {count:,} reads loaded")
    
    if not kmer_vectors:
        print("\nError: No sequences loaded. Check your FASTA file paths.")
        return 1
    
    # Convert to numpy arrays
    kmer_matrix = np.vstack(kmer_vectors)
    gc_array = np.array(gc_values)
    isolate_array = np.array(isolate_labels)
    
    print(f"\n  Total: {len(kmer_matrix):,} sequences from {len(fasta_files)} isolates")
    
    # --- Step 2 & 3: Clustering ---
    print("\n[2/4] Performing dimensionality reduction and clustering...")
    
    embeddings, cluster_ids = perform_clustering(kmer_matrix)
    if cluster_ids is None:
        return 1
    
    print(f"  Created {len(set(cluster_ids))} clusters")
    
    # --- Step 4: Build cluster summaries ---
    print("\n[3/4] Analyzing clusters...")
    
    cluster_summary = defaultdict(lambda: {
        "count": 0,
        "gc": [],
        "isolates": Counter(),
        "kmer_vectors": []
    })
    
    for cid, iso, gc, vec in zip(cluster_ids, isolate_array, gc_array, kmer_matrix):
        cluster_summary[cid]["count"] += 1
        cluster_summary[cid]["gc"].append(gc)
        cluster_summary[cid]["isolates"][iso] += 1
        cluster_summary[cid]["kmer_vectors"].append(vec)
    
    # Detect resistance motifs
    motif_counts = detect_resistance_motifs(kmer_matrix, cluster_ids, isolate_array)
    global_kmer_mean = np.mean(kmer_matrix, axis=0)
    
    # --- Step 5: Generate report ---
    print("\n[4/4] Generating report...")
    generate_cluster_report(cluster_summary, motif_counts, global_kmer_mean)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60 + "\n")
    
    return 0

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Parse command line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze WGS data for carbapenem resistance patterns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python carbapenem_resistance_analysis.py
    python carbapenem_resistance_analysis.py --fastas sample1=data/s1.fasta sample2=data/s2.fasta
        """
    )
    parser.add_argument(
        "--fastas", 
        nargs="*", 
        metavar="NAME=PATH",
        help="Isolate FASTA mappings (e.g., isolate1=path/to/file.fasta)"
    )
    
    args = parser.parse_args()
    
    # Use provided FASTA files or defaults
    fasta_files = ISOLATE_FASTAS.copy()
    if args.fastas:
        for pair in args.fastas:
            if "=" in pair:
                name, path = pair.split("=", 1)
                fasta_files[name] = path
            else:
                print(f"Warning: Invalid format '{pair}'. Use NAME=PATH")
    
    return run_analysis(fasta_files)

if __name__ == "__main__":
    sys.exit(main())