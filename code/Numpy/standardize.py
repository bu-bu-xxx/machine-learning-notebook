import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, X_input):
        X = X_input.copy()
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            X = X.to_numpy()
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
    
    def transform(self, X_input):
        X = X_input.copy()
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = (X - self.mean_) / self.std_
        if isinstance(X_input, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        return X
    
    def fit_transform(self, X_input):
        self.fit(X_input)
        return self.transform(X_input)

