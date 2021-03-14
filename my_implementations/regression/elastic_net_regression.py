import numpy as np


class ElasticNetRegressor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, etha=0.01, n_iter=500, tol=0.0001):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.etha = etha
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X_train, y_train):
        X_train = self.__get_addition_feature(X_train)
        n_samples = X_train.shape[0]
        self.w = np.zeros((X_train.shape[1]))

        for _ in range(self.n_iter):
            y_pred = X_train.dot(self.w)
            delta = y_pred - y_train
            gradient = X_train.T.dot(delta) + self.alpha * self.l1_ratio * np.sign(self.w) + \
                       self.alpha * (1 - self.l1_ratio) * np.sum(self.w)
            gradient /= n_samples
            old_w = self.w
            self.w -= self.etha * gradient

            if np.linalg.norm(old_w - self.w) < self.tol:
                break

    def __get_addition_feature(self, X):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=int)))

    def predict(self, X_test):
        X_test = self.__get_addition_feature(X_test)
        return X_test.dot(self.w)

    def rmse_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sqrt(np.mean((y_pred - y_test) ** 2))