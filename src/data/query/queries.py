import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import sys

# n_samples : Number of data points in dataset
# n_queries :Number of query points
# n_features : Dimensionality of each point
# k : Number of nearest neighbors to find
# seed : For reproducibility

if __name__ == "__main__":
    n_samples = int(sys.argv[1])   
    n_queries = int(sys.argv[2])      
    n_features = int(sys.argv[3])    
    k = int(sys.argv[4])           
    seed = int(sys.argv[5])         

    np.random.seed(seed)

    dataset = np.random.uniform(low=0.0, high=10.0, size=(n_samples, n_features))

    # Generate queries - perturb some dataset points slightly
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    queries = dataset[query_indices] + np.random.normal(0, 0.1, size=(n_queries, n_features))

    # Compute ground truth k-NN
    distances = euclidean_distances(queries, dataset)  # Shape: (n_queries, n_samples)
    ground_truth_indices = np.argsort(distances, axis=1)[:, :k]  # Top k indices per query

    dataset = dataset.astype(np.float64)
    queries = queries.astype(np.float64)
    ground_truth_indices = ground_truth_indices.astype(int)

    pd.DataFrame(dataset).to_csv("dataset.csv", index=True, header=True)
    pd.DataFrame(queries).to_csv("queries.csv", index=True, header=True)
    pd.DataFrame(ground_truth_indices).to_csv("ground_truth.csv", index=True, header=True)

    print(f"Generated dataset.csv: {n_samples} samples, {n_features} features")
    print(f"Generated queries.csv: {n_queries} queries, {n_features} features")
    print(f"Generated ground_truth.csv: {n_queries} queries, {k} nearest neighbors")