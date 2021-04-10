import numpy as np
from layers import Layer, InputLayer


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
                layer = Layer(neurons_num, self.input_layer.neurons_number,
                            act_func, inputs.shape[0])
            else:
                layer = Layer(neurons_num, self.layers[idx - 1].neurons_number,
                            act_func, inputs.shape[0])
            self.layers.append(layer)


    def _back_propagation(self, y):
        self.layers[-1].deltas = self.layers[-1].act_func_values - y
        for idx in range(len(self.layers[-2::-1])):
            layer, next_layer = self.layers[idx], self.layers[idx + 1]
            self.layers[idx].deltas = next_layer.dot(next_layer.deltas) * \
                        layer.activation_func_der(layer.sum_func_values)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                prev_layer_act_func_values = self.input_layer.act_func_values
            else:
                prev_layer_act_func_values = self.layers[idx - 1].act_func_values
            self.layers[idx].weights -= self.learning_rate * \
                layer.deltas.T * prev_layer_act_func_values
            self.layers[idx].biases -= self.learning_rate * layer.deltas


    def predict(self, X_test):
        output_layer = self.layers[-1]
        predictions = []
        for x in X_test:
            output_layer.calculate_act_func_values(x)
            predictions.append(output_layer.act_func_values)
        return np.array(predictions)

    def regression_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean((y_pred - y_test) ** 2)
