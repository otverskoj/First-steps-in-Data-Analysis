import numpy as np


class Node:
    def __init__(self, X, y,
                left_child=None, right_child=None, parent=None):
        self.feature_index = None
        self.threshold = None
        self.X = X
        self.y = y
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent

    def is_leaf(self):
        return not (self.right_child or self.left_child)


class DecisionTreeRegressor:
    def __init__(self, max_depth=1000, min_samples=5):
        self.root = None
        self.depth = 0
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X_train, y_train):
        self.feature_index, self.threshold = self.__calculate_best_index(X_train, y_train)
        X, y = np.delete(X_train, self.feature_index, axis=1), y_train
        self.root = Node(X, y)
        self.depth += 1

        X_left, X_right, y_left, y_right = self.__split_data(self.feature_index, self.threshold,
                                                            X, y, X_train)

        self.left_child = Node(X_left, y_left, parent=self.root)
        self.__make_tree(self.left_child)
        self.right_child = Node(X_right, y_right, parent=self.root)
        self.__make_tree(self.right_child)

    def __split_data(self, split_index, threshold, curr_X, curr_y, truly_X):
        X = truly_X[:, split_index]
        X_left = curr_X[X <= threshold]
        X_right = curr_X[X > threshold]
        y_left = curr_y[X <= threshold]
        y_right = curr_y[X > threshold]
        return X_left, X_right, y_left, y_right

    def __make_tree(self, current_node):
        if current_node.X.shape[0] <= self.min_samples or self.depth == self.max_depth \
            or current_node.X.shape[1] == 1:
            return

        current_node.feature_index, current_node.threshold = self.__calculate_best_index(current_node.X, current_node.y)
        if current_node.feature_index == 0:
            print(current_node.feature_index)
        X, y = np.delete(current_node.X, current_node.feature_index, axis=1), current_node.y
        X_left, X_right, y_left, y_right = self.__split_data(current_node.feature_index, current_node.threshold,
                                                            X, y, current_node.X)

        current_node.left_child = Node(X_left, y_left, parent=current_node)
        self.__make_tree(current_node.left_child)
        current_node.right_child = Node(X_right, y_right, parent=current_node)
        self.__make_tree(current_node.right_child)

        self.depth += 1

    def __calculate_best_index(self, X_train, y_train):
        best_index, best_t = None, None
        best_split_q = np.inf
        for feature_index in range(X_train.shape[1]):
            threshold = self.__calculate_best_threshold(X_train, y_train, feature_index)
            X, y = X_train[:, feature_index], y_train
            X_left = X[X <= threshold]
            X_right = X[X > threshold]
            y_left = y[X <= threshold]
            y_right = y[X > threshold]
            split_q = self.__calculate_q(X_left, X_right, y_left, y_right, X)
            if split_q <= best_split_q:
                best_index, best_t = feature_index, threshold

        return best_index, best_t

    def __calculate_q(self, X_l, X_r, y_l, y_r, X):
        left_loss = self.__loss_function(X_l, y_l)
        right_loss = self.__loss_function(X_r, y_r)
        left_norm = X_l.shape[0] / X.shape[0]
        right_norm = X_r.shape[0] / X.shape[0]
        q = left_norm * left_loss + right_norm * right_loss
        return q if not np.isnan(q) else np.inf

    def __loss_function(self, X, y):
        return np.sum((y - np.mean(y)) ** 2) / X.shape[0]
    
    def __calculate_best_threshold(self, X_train, y_train, feature_index):
        return np.mean(X_train[:, feature_index])

    def predict(self, X_test):
        y_pred = []
        for x_test in X_test:
            prediction = self.__predict(x_test, self.root)
            y_pred.append(prediction)

        return y_pred

    def __predict(self, x_test, current_node):
        if current_node.is_leaf():
            return np.mean(current_node.y)

        if x_test[current_node.feature_index] <= current_node.threshold:
            return self.__predict(x_test, current_node.left_child)
        else:
            return self.__predict(x_test, current_node.right_child)
        

    def rmse_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sqrt(np.mean((y_pred - y_test) ** 2))
