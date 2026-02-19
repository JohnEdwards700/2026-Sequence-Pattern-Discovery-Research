import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import Counter

def plot_cluster_distribution(cluster_ids):
    cluster_counts = Counter(cluster_ids)
    clusters = list(cluster_counts.keys())
    counts = list(cluster_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(clusters, counts, color='skyblue')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Sequences')
    plt.title('Cluster Distribution')
    plt.xticks(clusters)
    plt.show()

def calculate_silhouette_score(embeddings, cluster_ids):
    score = silhouette_score(embeddings, cluster_ids)
    print(f'Silhouette Score: {score:.4f}')
    return score

def summarize_clusters(cluster_summary):
    for cid, info in cluster_summary.items():
        mean_gc = np.mean(info["gc"])
        print(f"Cluster {cid}:")
        print(f"  Number of sequences: {info['count']}")
        print(f"  Mean GC content: {mean_gc:.3f}")
        print(f"  Isolate distribution: {dict(info['isolates'])}")
        print("")