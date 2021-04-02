import numpy as np


class KNearesrtNeighbors():

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predicted = []
        for idx in range(X_test.shape[0]):
            distances = self.__get_distances_and_labels(X_test[idx])
            sorted_distances = sorted(distances)
            k = self.n_neighbors
            min_distances = [x[1] for x in sorted_distances[:k]]
            max_count = max(min_distances.count(1), min_distances.count(0))
            predicted_label = 1 if max_count == min_distances.count(1) else 0
            predicted.append(predicted_label)

        return np.array(predicted)

    def __get_distances_and_labels(self, x_test):
        distances = []
        for idx in range(self.X_train.shape[0]):
            distance = np.sqrt(np.sum((self.X_train[idx] - x_test) ** 2))
            distances.append((distance, self.y_train[idx]))

        return distances

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
