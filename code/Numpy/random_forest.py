from Numpy.decision_tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


class RandomForestClassifier:
    """
    X, y accept only pandas DataFrame and Series.
    """

    def __init__(
        self, max_depth=3, max_features="sqrt", n_estimators=100, replacement=False, 
        n_samples=0.5,
        random_state=11
    ):
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.replacement = replacement
        self.random_state = random_state
        self.n_samples = n_samples

        self.n_trees = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            row_choice = np.random.choice(X.shape[0], int(self.n_samples*X.shape[0]))
            X_train = X.iloc[row_choice, :]
            y_train = y.iloc[row_choice]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, max_features=self.max_features
            )
            tree.fit(X_train, y_train)
            self.n_trees.append(tree)

    def predict(self, X: pd.DataFrame):
        preddictions = []
        for tree in self.n_trees:
            preddictions.append(tree.predict(X).to_numpy())
        # voting
        preddictions = np.vstack(preddictions).T
        pred_max = [np.argmax(np.bincount(row)) for row in preddictions]
        return pd.Series(pred_max, index=X.index)


if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    wine_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset", "wine", "wine.data"
    )
    wine_data = pd.read_csv(wine_path, header=None)
    columns = [
        "class",
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline",
    ]
    wine_data.columns = columns

    # split dataset into train set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        wine_data.iloc[:, 1:], wine_data.iloc[:, 0], test_size=0.2, random_state=11
    )
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    rf_clf = RandomForestClassifier(
        max_depth=2, max_features="sqrt", n_estimators=100, replacement=False
    )
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test, y_test)
    print("accuracy:", np.mean((y_pred == y_test).astype(int)))
