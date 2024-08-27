import numpy as np
import pandas as pd
from graphviz import Digraph
import os
from collections import Counter


class Node:
    def __init__(self, X: pd.DataFrame, y: pd.Series, clf, depth=0, max_features="all"):
        self.X = X
        self.y = y
        self.depth = depth
        self.left = None
        self.right = None
        self.father = None
        self.max_features = max_features

        self.gini_val = self.gini_impurity(self.y)
        self.is_leaf = False
        self.cls = self.y.value_counts().idxmax()
        self.samples = self.X.shape[0]
        self.y_labels = clf.y_labels  # all y_labels in y_train
        counter = Counter(self.y)
        self.values = [
            counter[label] for label in self.y_labels
        ]  # list: values of y_labels

        self.feature_name = None
        self.split_value = None
        self.id = 0

    def gini_impurity(self, y: pd.Series) -> float:
        return 1 - np.power(y.value_counts().values / y.shape[0], 2).sum()

    def gini_index(self, y1: pd.Series, y2: pd.Series) -> float:
        return (
            self.gini_impurity(y1) * y1.shape[0] + self.gini_impurity(y2) * y2.shape[0]
        ) / (y1.shape[0] + y2.shape[0])

    def best_split(self) -> tuple[str, float]:
        """
        return [feature_name, split_value]
        """
        gini_val_min = 1
        best_attr = None
        best_split_value = None
        if self.max_features == "all":
            considered_features = self.X.columns
        elif self.max_features == "sqrt":
            considered_features = np.random.choice(
                self.X.columns, int(np.sqrt(self.X.shape[1])), replace=False
            )
        else:
            raise ValueError("max_features should be 'all' or 'sqrt'")
        for attr in considered_features:
            for split_value in np.unique(self.X[attr]):
                y_left = self.y[self.X[attr] <= split_value]
                y_right = self.y[self.X[attr] > split_value]
                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue
                gini_val = self.gini_index(y_left, y_right)
                if gini_val < gini_val_min:
                    gini_val_min = gini_val
                    best_attr = attr
                    best_split_value = split_value
        self.feature_name = best_attr
        self.split_value = best_split_value
        return self.feature_name, self.split_value


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=4,
        purity_function="gini",
        max_features="all",
    ):
        # hyperparameters
        self.min_samples_split = min_samples_split
        self.purity_function = purity_function
        self.max_depth = max_depth
        # internal variables
        self.queue = []
        self.root = None  # tree root
        self.feature_names = None
        self.y_labels = None
        self.max_features = max_features

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        # init
        self.feature_names = X_train.columns
        self.y_labels = np.unique(y_train)
        self.root = Node(
            X_train, y_train, self, depth=0, max_features=self.max_features
        )
        self.queue.append(self.root)

        # build tree, bfs
        while len(self.queue) != 0:
            node = self.queue.pop(0)
            if node.depth is not None and node.depth == self.max_depth:
                node.is_leaf = True
            elif node.X.shape[0] < self.min_samples_split:
                node.is_leaf = True
            elif np.unique(node.y).shape[0] == 1:
                node.is_leaf = True
            else:
                pass
            # calculate gini impurity
            if self.purity_function != "gini":
                raise NotImplementedError("Only gini impurity is supported.")
            elif not node.is_leaf:
                attr, split_value = node.best_split()
                X_left = node.X[node.X[attr] <= split_value]
                X_right = node.X[node.X[attr] > split_value]
                y_left = node.y[node.X[attr] <= split_value]
                y_right = node.y[node.X[attr] > split_value]
                if X_left.shape[0] == 0 or X_right.shape[0] == 0:
                    node.is_leaf = True
                else:
                    node.left = Node(
                        X_left,
                        y_left,
                        self,
                        depth=node.depth + 1,
                        max_features=self.max_features,
                    )
                    node.right = Node(
                        X_right,
                        y_right,
                        self,
                        depth=node.depth + 1,
                        max_features=self.max_features,
                    )
                    node.left.father = node
                    node.right.father = node
                    node.left.id = node.id * 2 + 1
                    node.right.id = node.id * 2 + 2
                    self.queue.append(node.left)
                    self.queue.append(node.right)

            if node.is_leaf:
                pass

        # print("Tree is built.")

    def predict_one(self, x: pd.Series, node):
        if node.is_leaf:
            return node.cls
        if x.loc[node.feature_name] <= node.split_value:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X_test) -> pd.Series:
        y_pred = []
        for idx in X_test.index:
            node = self.root
            y_pred.append(self.predict_one(X_test.loc[idx], node))

        return pd.Series(y_pred, index=X_test.index)

    def show(self) -> str:
        dot = Digraph(comment="Tree")

        def dfs(node):
            text = ""
            if not node.is_leaf:
                text += f"{node.feature_name} <= {node.split_value}\n"
            text += f"gini: {node.gini_val:.3f}\n"
            text += f"samples: {node.samples}\n"
            text += f"values: {node.values}\n"
            text += f"class: {node.cls}\n"
            dot.node(str(node.id), text, shape="box", style="filled, rounded")
            if not node.is_leaf:
                dfs(node.left)
                dfs(node.right)
                dot.edge(str(node.id), str(node.left.id), "True")
                dot.edge(str(node.id), str(node.right.id), "False")

        dfs(self.root)
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "tree_tmp"
        )
        dot.render(path, format="png")
        path += ".png"
        print(f"Tree picture is saved in {path}")
        return path


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

    dt_clf = DecisionTreeClassifier(max_depth=3, max_features="sqrt")
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)
    print(f"Accuracy: {np.mean((y_pred == y_test).astype(int))}")
