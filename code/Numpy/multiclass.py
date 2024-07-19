import numpy as np
import copy


class OneVsRestClassifier:
    def __init__(self, estimator):
        self.estimator = estimator
        self.n_features = 0
        self.classes = None
        self.estimators_list = []

    def fit(self, X_train, y_train):
        # define variables
        _, n_features = X_train.shape
        classes = np.unique(y_train)
        self.n_features = n_features
        self.classes = classes
        # training for each class
        # 编码为f1,f2,...,fn classifiers
        # fi = 1 if (y_train == i) else 0
        for i in range(classes.shape[0]):
            y_train_i = (y_train == classes[i]).astype(int)
            estimator_i = copy.deepcopy(self.estimator)
            estimator_i.fit(X_train, y_train_i)
            self.estimators_list.append(estimator_i)

    def predict(self, X_test):
        samples_num = X_test.shape[0]
        OVR_pred = np.zeros((samples_num, self.classes.shape[0]))
        # predict for each class
        for i in range(self.classes.shape[0]):
            OVR_pred[:, i] = self.estimators_list[i].predict(X_test).astype(int) # m x k
        classes_dist = np.eye(self.classes.shape[0]).astype(int) # k x k

        # 计算每个类别的海明距离
        pred_dist = np.zeros((samples_num, self.classes.shape[0]))
        for i in range(self.classes.shape[0]):
            pred_dist[:, i] = np.sum((OVR_pred != classes_dist[i, :]).astype(int), axis=1)
        pred_max_idx = np.argmin(pred_dist, axis=1)
        pred_max_val = self.classes[pred_max_idx]
        return pred_max_val



