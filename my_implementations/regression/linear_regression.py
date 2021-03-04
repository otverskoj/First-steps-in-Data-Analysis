import numpy as np


class LinearRegressor():

    def __init__(self, etha=0.01, n_iter=200, tol=1e-3):
        self.etha = etha
        self.n_iter = n_iter
        self.tol = tol
        self.w = None

    def fit(self, X_train, y_train):
        X_train = self.__get_addition_feature(X_train)

        self.w = np.zeros((X_train.shape[1]))

        for _ in range(self.n_iter):
            l = X_train.shape[0]
            gradient = 2 * np.matmul(X_train.T, np.matmul(X_train, self.w) - y_train) / l 
            old_w = self.w
            self.w -= self.etha * gradient

            if np.linalg.norm(self.w - old_w) < self.tol:
                break

    def predict(self, X_test):
        X_test = self.__get_addition_feature(X_test)
        y_pred = []

        for x_test in X_test:
            prediction = np.sum(x_test * self.w.T)
            y_pred.append(prediction)

        return np.array(y_pred)

    def __get_addition_feature(self, X):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=int)))

    def mse_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean((y_pred - y_test) ** 2)
