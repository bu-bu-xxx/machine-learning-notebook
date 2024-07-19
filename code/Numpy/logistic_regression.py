import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self,
                 solver='newton-cg',
                 max_iter=100,
                 tol = 1e-4, # tolerance for stopping criterion
                 ):
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None
        self.mean = None
        self.std = None
        # self.multi_class = multi_class
        
    def _sigmoid(self, z) -> np.array:
        return 1 / (1 + np.exp(-z))
    
    def _loss_function(self, beta, X, y) -> float:
        z = X.dot(beta)
        loss = np.sum(-y * z + np.log(1 + np.exp(z)))
        return loss
    
    def _newton_cg(self, X, y, iters) -> np.array:
        p1 = lambda beta: self._sigmoid(X.dot(beta)) # m x 1
        derivative_1 = lambda beta: -np.sum(X.T * (y - p1(beta)), axis=1) # n x 1
        derivative_2 = lambda beta: \
        (X.T * p1(beta) * (1 - p1(beta))).dot(X) # n x n
        beta = np.random.rand(X.shape[1])
        for i in range(iters):
            beta_orig = beta
            beta = beta - np.linalg.pinv(derivative_2(beta)).dot(derivative_1(beta))
            if np.linalg.norm(beta - beta_orig) < self.tol:
                break
        return beta

    def _check_dimensions(self, X, y):
        if y.ndim > 1:
            raise ValueError('y must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same dimensions')
    
    def _scaled_X(self, X) -> np.array:
        # 不应该在算法里实现normalization，应该在数据处理阶段实现
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std
    
    def fit(self, X, y):
        self._check_dimensions(X, y)
        if self.solver == 'newton-cg':
            self.beta = self._newton_cg(X, y, self.max_iter)
        else:
            raise ValueError('Unsupported solver')
        
    def predict(self, X):
        if self.beta is None:
            raise ValueError('Model is not fitted yet')
        y_pred = self._sigmoid(X.dot(self.beta))
        y_pred = (y_pred > 0.5)
        return y_pred.astype(int)
    

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 20, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 1, 0, 0])
    model = LogisticRegression(solver='newton-cg', max_iter=1000, tol=1e-4)
    model._scaled_X(X)
    model.fit(X[:3, :], y[:3])
    # print(model.beta.shape)
    print(model._loss_function(model.beta, X, y))
    # print(model._loss_function(np.array([0, 1, 0]), X, y))
    print(model.beta)
    print(model.predict(X))
