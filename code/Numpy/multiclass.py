import numpy as np
import pandas as pd
import copy


class OneVsRestClassifier:
    def __init__(self, estimator):
        self.estimator = estimator
        self.reset_params()

    def reset_params(self):
        self.n_features = 0
        self.classes = None
        self.estimators_list = []
        # pandas variables
        self.features = None
        self.target_name = None
        self.train_index = None
        self.pred_index = None

    def fit(self, X:pd.DataFrame, y:pd.Series):
        self.reset_params()
        # check input
        if isinstance(X, pd.DataFrame):
            X_train = X.values
            self.features = X.columns
            self.train_index = X.index
        else:
            raise TypeError("X must be pandas.DataFrame")
        if isinstance(y, pd.Series):
            y_train = y.values
            self.target_name = y.name
        else:
            raise TypeError("y must be pandas.Series")

        # define variables
        self.n_features = X_train.shape[1]
        self.classes = np.unique(y_train)
        X_train = pd.DataFrame(X_train, columns=self.features, index=self.train_index)
        # training for each class
        # 编码为f1,f2,...,fn classifiers
        # fi = 1 if (y_train == i) else 0
        for i in range(self.classes.shape[0]):
            y_train_i = (y_train == self.classes[i])
            y_train_i = pd.Series(y_train_i, name=self.target_name, index=self.train_index)
            estimator_i = copy.deepcopy(self.estimator)
            estimator_i.fit(X_train, y_train_i)
            self.estimators_list.append(estimator_i)

    def predict(self, X):
        # check input
        if isinstance(X, pd.DataFrame):
            X_test = X.copy()
        else:
            raise TypeError("X must be pandas.DataFrame")

        samples_num = X_test.shape[0]
        OVR_pred = np.zeros((samples_num, self.classes.shape[0]))
        # predict for each class
        for i in range(self.classes.shape[0]):
            OVR_pred[:, i] = self.estimators_list[i].predict(X_test).to_numpy().astype(int) # m x k
        classes_dist = np.eye(self.classes.shape[0]) # k x k

        # 计算每个类别的海明距离
        pred_dist = np.zeros((samples_num, self.classes.shape[0]))
        for i in range(self.classes.shape[0]):
            pred_dist[:, i] = np.sum((OVR_pred != classes_dist[i, :]).astype(int), axis=1)
        pred_max_idx = np.argmin(pred_dist, axis=1)
        pred_max_val = self.classes[pred_max_idx]
        if isinstance(X, pd.DataFrame):
            pred_max_val = pd.Series(pred_max_val, name=self.target_name,
                                     index=self.pred_index)
        return pred_max_val



