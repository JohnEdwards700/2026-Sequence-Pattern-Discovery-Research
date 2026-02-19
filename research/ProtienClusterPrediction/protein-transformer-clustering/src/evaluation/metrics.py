from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def calculate_silhouette_score(embeddings, cluster_labels):
    if len(set(cluster_labels)) > 1:  # At least 2 clusters are needed
        return silhouette_score(embeddings, cluster_labels)
    return None

def calculate_adjusted_rand_index(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

def calculate_normalized_mutual_info(true_labels, predicted_labels):
    return normalized_mutual_info_score(true_labels, predicted_labels)