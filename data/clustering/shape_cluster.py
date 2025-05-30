from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def visualize_clusters(X, labels_true, labels_pred, title_true="Ground Truth", title_pred="DBSCAN Clustering"):
    """
    Visualize the ground truth and predicted clusters side by side.
    
    Parameters:
    X : array-like, shape (n_samples, 2)
        The input data for plotting (2D features).
    labels_true : array-like, shape (n_samples,)
        True cluster labels.
    labels_pred : array-like, shape (n_samples,)
        Predicted cluster labels from DBSCAN.
    title_true : str
        Title for the ground truth plot.
    title_pred : str
        Title for the predicted clusters plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth plot
    unique_labels_true = np.unique(labels_true)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels_true)))
    for k, col in zip(unique_labels_true, colors):
        mask = labels_true == k
        ax1.scatter(X[mask, 0], X[mask, 1], c=[col], label=f'Cluster {k}', s=50)
    ax1.set_title(title_true)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()

    # DBSCAN prediction plot
    unique_labels_pred = np.unique(labels_pred)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels_pred)))
    for k, col in zip(unique_labels_pred, colors):
        if k == -1:
            # Black used for noise points
            col = [0, 0, 0, 1]
            label = 'Noise'
        else:
            label = f'Cluster {k}'
        mask = labels_pred == k
        ax2.scatter(X[mask, 0], X[mask, 1], c=[col], label=label, s=50)
    ax2.set_title(title_pred)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    random_state = 42
    moons, y1 = datasets.make_moons(n_samples=1500, noise=0.0781, random_state=random_state)
    blobs, y2 = datasets.make_blobs(n_samples=1500, centers=[(-0.85, 2.75), (1.75, 2.25)], 
                                    cluster_std=0.5231, random_state=random_state)
    X = np.vstack([moons, blobs])

    # Fix the label offset: moons labels are {0, 1}, blobs labels become {2, 3}
    y2 = y2 + len(np.unique(y1))  # Offset blobs labels by 2
    labels_true = np.hstack([y1, y2])

    # Standardize features
    X = StandardScaler().fit_transform(X)
    X_new = np.hstack((X, labels_true.reshape(-1, 1)))
    X_new = pd.DataFrame(X_new, columns=['Feature1', 'Feature2', 'Label'])
    pd.DataFrame(X_new).to_csv("shape_clusters_include_y.csv", index=True, header=True)

    # Apply DBSCAN
    db = DBSCAN(eps=0.1, min_samples=10).fit(X)
    labels = db.labels_

    # Print clustering metrics
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
    print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

    # Visualize the results
    visualize_clusters(X, labels_true, labels)