from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":

    diabetes = load_diabetes()
    X = diabetes.data

    y = diabetes.target

    X_new = np.hstack((X, y.reshape(-1, 1)))
    X_new = pd.DataFrame(X_new, columns=list(diabetes.feature_names)+['label'])
    X_new.to_csv("diabetes.csv")


    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

    regressor.fit(X[:300], y[:300])
    predictions = regressor.predict(X[300:])

    mse = mean_squared_error(y[300:], predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(y[300:], predictions)
    print(f'R-squared: {r2}')

    X = pd.DataFrame(X, columns=diabetes.feature_names)
    X.to_csv("diabetes_features.csv")
