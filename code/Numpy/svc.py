import numpy as np
import pandas as pd


class SVC:
    """
    soft-margin support vector classifier
    objective function:
        min R + C*L
        Regularization = 1/2 * ||w||^2
        Loss = sum[max(0, 1-yi(wxi))]
    """
    def __init__(self, kernel='linear', C=1.0, loss='hinge'):
        # hyperparameters
        self.kernel = kernel
        self.C = C # scalar
        self.loss = loss

        self.reset_parameters()
        # check
        if self.kernel not in ['linear']:
            raise TypeError(f"kernel does not support: {self.kernel}")
        if self.loss not in ['hinge']:
            raise TypeError(f"loss does not support: {self.loss}")
    
    def reset_parameters(self):
        # model parameters
        self.w = None # d
        self.b = None # scalar
        self.X_train = None # temp m*d DataFrame
        self.y_train = None # temp m Series
        self.alpha = None # m 
        self.columns = None # d X_train.columns
        self.target_name = None # y_train.name 
        self.label = None # 2 y_train.unique()
        self.tol = min(1e-3, 1/self.C**2) # tolerance for number closed to zero and stopping criterion
        self.max_iter = 1000 # maximum number of iterations
        self.max_passes = 10 # maximum number of passes

    def kernel_func(self, X1, X2) -> np.ndarray:
        """
        X1: m*d 
        X2: test_size*d
        return: m*test_size
        """
        if self.kernel == 'linear':
            return X1.dot(X2.T)
    
    def loss_func(self) -> np.ndarray: 
        """
        return: m dim
        """
        return np.maximum(0, 1 - self.y_train * \
                          (self.X_train.dot(self.w.reshape(-1, 1)).reshape(-1) + self.b))

    def regularization(self) -> float:
        return 0.5 * np.sum(self.w**2)
    
    def model_func(self, X) -> np.ndarray:
        """
        f(x) = w^T*x + b
        return: m dim f(x)
        """
        X_test = X.copy().to_numpy()
        y_pred = X_test.dot(self.w.reshape(-1, 1)).reshape(-1) + self.b
        # y_pred = pd.Series(y_pred, index=X.index, name=self.target_name)
        # y_pred[y_pred >= 0] = self.label[0]
        # y_pred[y_pred < 0] = self.label[1]
        return y_pred

    def dual_func(self, alpha: np.ndarray=None) -> float:
        if alpha is None:
            alpha = self.alpha
        tmp = alpha.reshape(-1, 1)*self.y_train.reshape(-1, 1)*self.X_train
        return np.sum(alpha) - 0.5 * np.sum(tmp.dot(tmp.T))

    def primal_func(self) -> float:
        return self.regularization() + self.C * np.sum(self.loss_func())
    
    def w_update(self) -> np.ndarray:
        if self.kernel == 'linear':
            return np.sum((self.alpha*self.y_train).reshape(-1, 1)*self.X_train, axis=0)

    def fit(self, X:pd.DataFrame, y:pd.Series):
        self.reset_parameters()
        X_train, y_train = X.copy(), y.copy()
        self.preprocess(X_train, y_train)
        self.smo()
        self.w = self.w_update()

    def predict(self, X:pd.DataFrame) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be pandas.DataFrame")
        X_test = X.copy()
        X_test = X_test.to_numpy() 
        # # X_test = X_train
        # ww = (self.alpha*self.y_train).reshape(-1, 1)*self.X_train
        # # wwxx = np.sum(self.kernel_func(X_train, X_test), axis=0)
        # wwxx = np.sum(self.kernel_func(self.X_train, X_test), axis=0)
        # print(f"ww: {wwxx}")
        # print("-------------------")
        # print(self.alpha.reshape(-1, 1)*self.y_train.reshape(-1, 1))
        y_pred =  np.sum(self.alpha.reshape(-1, 1)*self.y_train.reshape(-1, 1) \
                      * self.kernel_func(self.X_train, X_test), axis=0) + self.b
        y_pred = pd.Series(y_pred, index=X.index, name=self.target_name)
        y_pred[y_pred >= 0] = self.label[0]
        y_pred[y_pred < 0] = self.label[1]
        return y_pred

    def b_func(self) -> float:
        return np.mean(self.y_train - np.sum(
            self.alpha.reshape(-1, 1)* self.y_train.reshape(-1, 1)* \
                self.kernel_func(self.X_train, self.X_train), axis=0
            ).reshape(-1))

    def check_KKT(self, alpha, epsilon) -> bool:
        """
        alpha: self.alpha[i]
        epsilon: 1-yi*f(xi) = -Ei*yi
        check KKT conditions
        """
        if (self.tol < alpha < (self.C - self.tol)) and (np.abs(epsilon) <= self.tol):
            return True
        if (alpha <= self.tol) and (epsilon < self.tol):
            return True
        if (alpha >= (self.C - self.tol)) and (epsilon > -self.tol):
            return True
        return False

    def smo(self):
        """
        return: alpha m*1, b scalar
        """
        # initialize alpha, b
        np.random.seed(0)
        m = self.X_train.shape[0]
        # self.alpha = np.ones(m) * self.C / 2 # 用这个初始值，当C很大时容易数值爆炸
        self.alpha = np.zeros(m)
        self.b = self.b_func()
        # loop
        iter_cnt = 0
        passes = 0
        E_func = lambda k: self.f_func(k)-self.y_train[k]
        k_func = lambda i, j: self.kernel_func(self.X_train[i].reshape(1, -1), self.X_train[j].reshape(1, -1))[0, 0]
        eta_func = lambda i, j: 2*k_func(i, j) - k_func(i, i) - k_func(j, j)
        def LH_func(i, j, ai, aj) -> list[float, float]:
            if self.y_train[i] != self.y_train[j]:
                L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
            else:
                L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
            return L, H
        
        while iter_cnt < self.max_iter and passes < self.max_passes:
            tag_pass = False
            for i in np.random.permutation(m):
                if self.check_KKT(self.alpha[i], 1-self.y_train[i]*self.f_func(i)):
                    continue
                for j in np.random.permutation(m):
                    # calculate
                    ai, aj = self.alpha[i], self.alpha[j]
                    Ei, Ej = E_func(i), E_func(j)
                    eta = eta_func(i, j)
                    L, H = LH_func(i, j, ai, aj)
                    if (np.abs(eta) < self.tol) or (np.abs(L - H) < self.tol):
                        continue
                    # clip ai, aj
                    aj_new = aj - self.y_train[j]*(Ei - Ej)/eta
                    aj_clip = np.clip(aj_new, L, H)
                    ai_clip = ai + self.y_train[i]*self.y_train[j]*(aj - aj_clip)
                    if (np.abs(ai_clip - ai) + np.abs(aj_clip - aj)) < self.tol:
                        continue
                    # update alpha, b
                    self.alpha[i], self.alpha[j] = ai_clip, aj_clip
                    bi = self.b - Ei - self.y_train[i]*(ai_clip - ai)*k_func(i, i) - \
                        self.y_train[j]*(aj_clip - aj)*k_func(i, j)
                    bj = self.b - Ej - self.y_train[i]*(ai_clip - ai)*k_func(i, j) - \
                        self.y_train[j]*(aj_clip - aj)*k_func(j, j)
                    if 0 < ai_clip < self.C:
                        self.b = bi
                    elif 0 < aj_clip < self.C:
                        self.b = bj
                    else:
                        self.b = (bi + bj) / 2
                    tag_pass = True

            if not tag_pass:
                passes += 1
            iter_cnt += 1
        return

    def f_func(self, k) -> float:
        return np.sum(self.alpha.reshape(-1, 1)*self.y_train.reshape(-1, 1)* \
            self.kernel_func(self.X_train, self.X_train[k].reshape(1, -1))) + self.b
    
    def preprocess(self, X_train:pd.DataFrame, y_train:pd.Series):
        """
        standardize X_train, substitute y_train with {-1, 1}
        return: X_train, y_train (values in {-1, 1})
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X must be pandas.DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y must be pandas.Series")
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.label = np.unique(self.y_train)
        if self.label.shape[0] != 2:
            raise ValueError("only binary classification is supported, please check y")
        self.y_train[self.y_train == self.label[0]] = 1
        self.y_train[self.y_train == self.label[1]] = -1
        self.target_name = y_train.name
        self.columns = X_train.columns


if __name__ == '__main__':
    # debug finished
    # preprocess the data
    import numpy as np
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    from standardize import StandardScaler


    # load the data
    wine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'dataset', 'wine', 'wine.data')
    wine_data = pd.read_csv(wine_path, header=None, sep=',')
    attributes = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    wine_data.columns = attributes
    wine_data_12 = wine_data[wine_data.loc[:, 'Class label'] != 1]
    wine_data_12 = wine_data_12.reset_index(drop=True)

    X = wine_data_12.iloc[:, 1:]
    y = wine_data_12.iloc[:, 0]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # standardize the data



    # train the model
    svc_clf = SVC(kernel='linear', C=1e4)
    svc_clf.fit(X_train, y_train)

    y_pred = svc_clf.predict(X_train)
    print((y_pred == y_train).value_counts())
    aa = svc_clf.model_func(X_train)
    print(aa[aa < 0])
    print("---------------------------")
    # print(svc_clf.model_func())

    # print(y_pred)
    # print(y_train)


    # max_val = -float('inf')
    # for _ in range(10000):
    #     alpha = np.random.rand(svc_clf.X_train.shape[0])
    #     max_val = max(max_val, svc_clf.dual_func(alpha))
    # print(max_val)

    # print(svc_clf.dual_func())
    # print(svc_clf.primal_func())
    # print(svc_clf.alpha)


    from sklearn.svm import LinearSVC

    linear = LinearSVC(C=1e4)
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_train)
    print((y_pred == y_train).value_counts())
    ff = linear.decision_function(X_train)
    print(ff[ff < 0])

    print(svc_clf.b)
    print(linear.intercept_)
