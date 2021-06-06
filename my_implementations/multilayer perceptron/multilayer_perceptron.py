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
        for sample_idx in range(X_train.shape[0]):
            self._forward_propagation(X_train[sample_idx])
            self._back_propagation(y_train[sample_idx])

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
        for idx in range(len(self.layers) - 2, -1, -1):
            layer, next_layer = self.layers[idx], self.layers[idx + 1]
            self.layers[idx].deltas = np.dot(next_layer.deltas, next_layer.weights.T) * \
                        layer.activation_func_der(layer.sum_func_values)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                prev_layer = self.input_layer
            else:
                prev_layer = self.layers[idx - 1]
            prev_layer_act_func_values = prev_layer.act_func_values
            self.layers[idx].weights -= self.learning_rate * \
                layer.deltas * prev_layer_act_func_values.reshape(
                    (prev_layer.neurons_number, 1))
            self.layers[idx].biases -= self.learning_rate * layer.deltas


    def predict(self, X_test):
        predictions = []
        for sample_idx in range(X_test.shape[0]):
            length = X_test[sample_idx].shape[0]
            test_sample = X_test[sample_idx].reshape(1, length)
            self.input_layer.calculate_act_func_values(test_sample)
            for idx in range(len(self.layers)):
                if idx == 0:
                    inputs = self.input_layer.act_func_values
                else:
                    inputs = self.layers[idx - 1].act_func_values
                self.layers[idx].calculate_act_func_values(inputs)
            predictions.append(self.layers[-1].act_func_values)
        
        # output_layer = self.layers[-1]
        # predictions = []
        # for sample_idx in range(X_test.shape[0]):
        #     length = X_test[sample_idx].shape[0]
        #     test_sample = X_test[sample_idx].reshape(1, length)
        #     output_layer.calculate_act_func_values(test_sample)
        #     predictions.append(output_layer.act_func_values)
        return np.array(predictions)

    def regression_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean((y_pred - y_test) ** 2)
