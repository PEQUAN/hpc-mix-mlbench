import pandas as pd
import sys
from sklearn.datasets import make_blobs
import os
import numpy as np


if __name__ == "__main__":
    # if not os.path.isdir('./blobs'):
    #    os.mkdir('./blobs') 
    #else:
    #    print("Folder already existed")

    num = int(sys.argv[1])
    dim = int(sys.argv[2])
    n_clusters = int(sys.argv[3])

    X, y = make_blobs(n_samples=num, centers=n_clusters, n_features=dim, random_state=0)

    print(f"write {num} {dim}-data of {n_clusters} clusters:")
    print("X:\n", X[:5])
    print("y:\n", y[:5])
    #pd.DataFrame(X).to_csv(f"X_{dim}d_{n_clusters}.csv", index=False, header=False)
    #pd.DataFrame(y).to_csv(f"y_{dim}d_{n_clusters}.csv", index=False, header=False)
    
    X_new = np.hstack((X, y.reshape(-1, 1)))
    X_new = pd.DataFrame(X_new)
    pd.DataFrame(X_new).to_csv(f"blobs_{dim}d_{n_clusters}_include_y.csv", index=True, header=True)
    
