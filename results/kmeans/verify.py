import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami
# from sklearn.metrics import silhouette_score

def compute_sse(dataset_file, labels_file, centroids_file):
    try:
        # Read the dataset
        dataset = pd.read_csv(dataset_file, header=None)
        data_points = dataset.values  # Convert to numpy array for easier manipulation
        num_points, num_features = data_points.shape

        # Read the assigned labels
        labels_df = pd.read_csv(labels_file)
        labels = labels_df['cluster_label'].values
        if len(labels) != num_points:
            raise ValueError(f"Number of labels ({len(labels)}) does not match number of data points ({num_points})")

        # Read the centroids
        centroids_df = pd.read_csv(centroids_file)
        # centroid_indices = centroids_df['centroid_index'].values
        centroids = centroids_df.drop('centroid_index', axis=1).values  # Drop the index column
        num_centroids, centroid_features = centroids.shape

        # Verify dimensions
        if centroid_features != num_features:
            raise ValueError(f"Number of features in centroids ({centroid_features}) does not match dataset ({num_features})")
        if num_centroids != len(np.unique(labels)):
            print(f"Warning: Number of centroids ({num_centroids}) does not match number of unique labels ({len(np.unique(labels))})")

        # Compute SSE
        sse = 0.0
        for i in range(num_points):
            point = data_points[i]
            cluster_id = labels[i]
            if cluster_id >= num_centroids:
                print(f"Warning: Point {i} assigned to invalid cluster {cluster_id}, skipping...")
                continue
            centroid = centroids[cluster_id]
            # Compute squared Euclidean distance
            distance = np.sum((point - centroid) ** 2)
            sse += distance

        return sse

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    

    
if __name__ == "__main__":
    results = pd.read_csv("output_labels.csv")
    truth = pd.read_csv("../../data/blobs/y_2d_10.csv", header=None)
    results = results.set_index('point_index')

    print("AMI:", ami(results.cluster_label, np.array(truth).flatten()))
    print("ARI:", ari(results.cluster_label, np.array(truth).flatten()))

    dataset_file = "../../data/blobs/X_2d_10.csv"
    labels_file = "output_labels.csv"
    centroids_file = "centroids.csv"

    sse = compute_sse(dataset_file, labels_file, centroids_file)
    
    if sse is not None:
        print(f"Sum of Squared Errors (SSE): {sse}")
    else:
        print("Failed to compute SSE due to errors.")
