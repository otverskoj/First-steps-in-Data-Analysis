import numpy as np


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    @staticmethod
    def tanh(x):
        return 2 * ActivationFunctions.sigmoid(2 * x) - 1

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def rectified_linear_unit(x):
        return np.max(0, x)


class ActivationFunctionsDerivatives:
    @staticmethod
    def find_derivative(func):
        if func is ActivationFunctions.sigmoid:
            return ActivationFunctionsDerivatives.sigmoid_der
        elif func is ActivationFunctions.tanh:
            return ActivationFunctionsDerivatives.tanh_der
        elif func is ActivationFunctions.softplus:
            return ActivationFunctionsDerivatives.softplus_der
        elif func is ActivationFunctions.rectified_linear_unit:
            return ActivationFunctionsDerivatives.rectified_linear_unit_der

    @staticmethod
    def sigmoid_der(x):
        return ActivationFunctions.sigmoid(x) * \
            (1 - ActivationFunctions.sigmoid(x))

    @staticmethod
    def tanh_der(x):
        return 4 * ActivationFunctionsDerivatives.sigmoid_der(2 * x)

    @staticmethod
    def softplus_der(x):
        return 1 - ActivationFunctions.sigmoid(-1 * x)

    @staticmethod
    def rectified_linear_unit_der(x):
        x_der = np.array(x)
        x_der[x_der <= 0] = 0
        x_der[x_der > 0] = 1
        return x_der