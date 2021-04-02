import numpy as np


class LinearSVM():

    def __init__(self, etha=0.01, alpha=0.1, n_iter=200):
        self.etha = etha
        self.alpha = alpha
        self.n_iter = n_iter
        self.w = None

    def fit(self, X_train, y_train):
        X_train = self.__get_addition_feature(X_train)
        # подразумевается, что решается задача бинирной классификации
        label_0, label_1 = np.unique(y_train)
        y_train[y_train == label_0] = -1
        y_train[y_train == label_1] = 1

        self.w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])

        for _ in range(self.n_iter):
            for idx in range(X_train.shape[0]):
                margin = y_train[idx] * np.dot(self.w, X_train[idx])

                if margin >= 1:
                    gradient = self.alpha * self.w / self.n_iter # Зачем делить на n_iter?
                    self.w -= self.etha * gradient
                else:
                    gradient = self.alpha * self.w / self.n_iter - y_train[idx] * X_train[idx]
                    self.w -= self.etha * gradient

    def predict(self, X_test):
        X_test = self.__get_addition_feature(X_test)
        y_pred = []

        for idx in range(X_test.shape[0]):
            sign = np.sign(np.dot(self.w, X_test[idx]))
            prediction = 1 if sign >= 0 else 0
            y_pred.append(prediction)

        return np.array(y_pred)

    def __get_addition_feature(self, X):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=int)))

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
