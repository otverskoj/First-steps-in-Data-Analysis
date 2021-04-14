import numpy as np
from activation_functions import ActivationFunction, ActivationFunctionDerivative


class Layer:
    def __init__(self, neurons_num, prev_layer_neurons_num, activation_func, inputs_size):
        self.neurons_number = neurons_num
        self.prev_layer_neurons_num = prev_layer_neurons_num
        self.activation_func = activation_func
        self.activation_func_der = \
            ActivationFunctionDerivative.find_derivative(activation_func)
        self._setup_params(inputs_size)

    def _setup_params(self, inputs_size):
        loc_w, scale_w = 0, 1 / np.sqrt(inputs_size)
        self.weights = np.random.normal(loc_w, scale_w,
                        (self.prev_layer_neurons_num, self.neurons_number))
        loc_b, scale_b = 0, 1
        self.biases = np.random.normal(loc_b, scale_b, (1, self.neurons_number))
        self.deltas = None

    def calculate_act_func_values(self, inputs):
        self.sum_func_values = np.dot(inputs, self.weights) + self.biases
        self.act_func_values = self.activation_func(self.sum_func_values)


class InputLayer:
    def __init__(self, inputs):
        self.neurons_number = inputs.shape[0]

    def calculate_act_func_values(self, inputs):
        self.act_func_values = inputs