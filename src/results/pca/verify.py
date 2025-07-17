import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

if __name__ == "__main__":

    X = pd.read_csv("../../data/blobs/X_20d_10.csv", header=None)
    print(X[:10])
    pca = PCA(n_components=3)
    projected = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(projected[:10])