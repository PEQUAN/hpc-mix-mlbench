import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

if __name__ == "__main__":
    # Parameters
    n_samples = 1000    # Number of data points in dataset
    n_features = 10    # Dimensionality of each point
    n_queries = 50      # Number of query points
    k = 3              # Number of nearest neighbors to find
    seed = 42          # For reproducibility

    # Set random seed
    np.random.seed(seed)

    # Generate dataset
    dataset = np.random.uniform(low=0.0, high=10.0, size=(n_samples, n_features))

    # Generate queries (perturb some dataset points slightly)
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    queries = dataset[query_indices] + np.random.normal(0, 0.1, size=(n_queries, n_features))

    # Compute ground truth k-NN
    distances = euclidean_distances(queries, dataset)  # Shape: (n_queries, n_samples)
    ground_truth_indices = np.argsort(distances, axis=1)[:, :k]  # Top k indices per query

    # Convert to float32 or float64 (choose one based on your C++ precision)
    # Use float64 (double) by default; comment/uncomment for float32 (single)
    dataset = dataset.astype(np.float64)
    queries = queries.astype(np.float64)
    ground_truth_indices = ground_truth_indices.astype(int)

    # Alternative: Use float32
    # dataset = dataset.astype(np.float32)
    # queries = queries.astype(np.float32)

    # Write to CSV files
    pd.DataFrame(dataset).to_csv("query/dataset.csv", index=False, header=False)
    pd.DataFrame(queries).to_csv("query/queries.csv", index=False, header=False)
    pd.DataFrame(ground_truth_indices).to_csv("query/ground_truth.csv", index=False, header=False)

    print(f"Generated dataset.csv: {n_samples} samples, {n_features} features")
    print(f"Generated queries.csv: {n_queries} queries, {n_features} features")
    print(f"Generated ground_truth.csv: {n_queries} queries, {k} nearest neighbors")