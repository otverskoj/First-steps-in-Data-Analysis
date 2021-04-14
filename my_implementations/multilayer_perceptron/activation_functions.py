import numpy as np


class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    @staticmethod
    def tanh(x):
        return 2 * ActivationFunction.sigmoid(2 * x) - 1

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def rectified_linear_unit(x):
        return np.max(0, x)


class ActivationFunctionDerivative:
    @staticmethod
    def find_derivative(func):
        if func is ActivationFunction.sigmoid:
            return ActivationFunctionDerivative.sigmoid_der
        elif func is ActivationFunction.tanh:
            return ActivationFunctionDerivative.tanh_der
        elif func is ActivationFunction.softplus:
            return ActivationFunctionDerivative.softplus_der
        elif func is ActivationFunction.rectified_linear_unit:
            return ActivationFunctionDerivative.rectified_linear_unit_der

    @staticmethod
    def sigmoid_der(x):
        return ActivationFunction.sigmoid(x) * \
            (1 - ActivationFunction.sigmoid(x))

    @staticmethod
    def tanh_der(x):
        return 4 * ActivationFunctionDerivative.sigmoid_der(2 * x)

    @staticmethod
    def softplus_der(x):
        return 1 - ActivationFunction.sigmoid(-1 * x)

    @staticmethod
    def rectified_linear_unit_der(x):
        x_der = np.array(x)
        x_der[x_der <= 0] = 0
        x_der[x_der > 0] = 1
        return x_der