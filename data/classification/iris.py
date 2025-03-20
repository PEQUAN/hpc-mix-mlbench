import pandas as pd
from sklearn import datasets
import numpy as np


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data

    y = iris.target

    X_new = np.hstack((X, y.reshape(-1, 1)))
    X_new = pd.DataFrame(X_new, columns=list(iris.feature_names)+['label'])
    X_new.to_csv("iris.csv")