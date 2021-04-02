import numpy as np


class GaussianNaiveBayes():

    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.prior_probas = self.__get_prior_probas(y_train)
        self.expected_values = self.__get_expected_values(X_train)
        self.variances = self.__get_variances(X_train, self.expected_values)
        self.gauss_multipliers = self.__get_gauss_multipliers(self.variances)

    def __get_prior_probas(self, y_train):
        dataset_length = y_train.shape[0]
        classes_proba = []
        for class_label in np.unique(y_train):
            class_length = y_train[y_train == class_label].shape[0]
            class_proba = class_length / dataset_length
            classes_proba.append((class_proba, class_label))

        return classes_proba

    def __get_expected_values(self, X_train):
        expected_values = []

        for features_idx in range(X_train.shape[1]):
            feature_values = X_train[:, features_idx]
            expected_values.append(np.mean(feature_values))

        return np.array(expected_values)

    def __get_variances(self, X_train, expected_values):
        variances = []

        for features_idx in range(X_train.shape[1]):
            feature_values = X_train[:, features_idx]
            expected_value = expected_values[features_idx]
            variance = np.mean( (feature_values - expected_value) ** 2 )
            variances.append(variance)

        return np.array(variances)

    def __get_gauss_multipliers(self, variances):
        gauss_multipliers = []

        for variance in variances:
            gauss_multiplier = 1.0 / np.sqrt(2 * np.pi * variance)
            gauss_multipliers.append(gauss_multiplier)

        return np.array(gauss_multipliers)

    def predict(self, X_test):
        predicted = []

        for x in X_test:            
            first_mult = np.prod( self.gauss_multipliers )
            second_mult = np.prod( np.exp( -1 * ((x - self.expected_values) ** 2) / (2 * self.variances) ) )
            cond_proba = first_mult * second_mult
            
            max_proba = -1
            for class_ in self.prior_probas:
                class_proba = class_[0] * cond_proba
                if class_proba > max_proba:
                    max_proba = class_proba
                    label_max_proba = class_[1]


            predicted.append(label_max_proba)

        return np.array(predicted)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
