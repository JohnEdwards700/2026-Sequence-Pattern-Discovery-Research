import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(embeddings, labels, title="t-SNE (2D)", random_state=42, perplexity=30):
    """
    embeddings: (n_samples, n_features) numpy array (use PCA embeddings from testfile.py)
    labels: list/array of isolate names for coloring
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity
    )
    X2 = tsne.fit_transform(embeddings)

    plt.figure()
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(X2[idx, 0], X2[idx, 1], s=10, label=str(lab))
    plt.legend()
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

    return X2