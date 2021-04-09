import numpy as np


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class ActivationFunctionsDerivatives:
    @staticmethod
    def find_derivative(func):
        if func is ActivationFunctions.sigmoid:
            return ActivationFunctionsDerivatives.sigmoid_der

    @staticmethod
    def sigmoid_der(x):
        return ActivationFunctions.sigmoid(x) * \
            (1 - ActivationFunctions.sigmoid(x))


class Layer:
    def __init__(self, neurons_num, prev_layer_neurons_num, activation_func):
        self.neurons_number = neurons_num
        self.prev_layer_neurons_num = prev_layer_neurons_num
        self.activation_func = activation_func
        self.activation_func_der = \
            ActivationFunctionsDerivatives.find_derivative(activation_func)
        self._setup_params()

    def _setup_params(self):
        self.weights = [[0 for _ in range(self.neurons_number)]
                        for _ in range(self.prev_layer_neurons_num)]
        self.biases = [0 for _ in range(self.neurons_number)]
        self.deltas = None

    def calculate_act_func_values(self, inputs):
        self.sum_func_values = self.weights.T.dot(inputs) + self.biases
        self.act_func_values = self.activation_func(self.sum_func_values)


class InputLayer:
    def __init__(self, inputs):
        self.neurons_number = inputs.shape[1]

    def calculate_act_func_values(self, inputs):
        self.act_func_values = inputs


class MultilayerPerceptron:
    def __init__(self, learning_rate, max_epochs_num, layers_params):
        self.learning_rate = learning_rate
        self.max_epochs_num = max_epochs_num
        self.layers_params = self._parse_layers_params(layers_params)
        self.layers = None

    def _parse_layers_params(self, layers_params):
        return layers_params

    def fit(self, X_train, y_train):
        for x, y in zip(X_train, y_train):
            self._forward_propagation(x)
            self._back_propagation(y)

    def _forward_propagation(self, x):
        if self.layers is None:
            self._build_layers(inputs=x)
        self.input_layer.calculate_act_func_values(x)
        for idx in range(len(self.layers)):
            if idx == 0:
                inputs = self.input_layer.act_func_values
            else:
                inputs = self.layers[idx - 1].act_func_values
            self.layers[idx].calculate_act_func_values(inputs)

    def _build_layers(self, inputs):
        self.input_layer = InputLayer(inputs)
        self.layers = []
        for idx, layer_params in enumerate(self.layers_params):
            neurons_num, act_func = layer_params
            if len(self.layers) == 0:
                layer = Layer(neurons_num, self.input_layer.neurons_number, act_func)
            else:
                layer = Layer(neurons_num, self.layers[idx - 1].neurons_number, act_func)
            self.layers.append(layer)


    def _back_propagation(self, y):
        self.layers[-1].deltas = self.layers[-1].act_func_values - y
        for idx in range(len(self.layers[-2::-1])):
            next_layer = self.layers[idx + 1]
            layer = self.layers[idx]
            self.layers[idx].deltas = \
                next_layer.dot(next_layer.deltas) * \
                layer.activation_func_der(layer.sum_func_values)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                prev_layer_act_func_values = self.input_layer.act_func_values
            else:
                prev_layer_act_func_values = self.layers[idx - 1].act_func_values
            self.layers[idx].weights -= self.learning_rate * \
                layer.deltas.T * prev_layer_act_func_values
            self.layers[idx].biases -= self.learning_rate * layer.deltas


    def predict(self):
        pass

    def score(self):
        pass
