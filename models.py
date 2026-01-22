import numpy as np

from supplementary import Value


def weight_init_function(layer_size1: int, layer_size2: int, x):
    return np.random.uniform(-1, 1, (layer_size1, layer_size2))


def bias_init_function(layer_size: int, x):
    return np.random.uniform(-1, 1, layer_size)


class NeuralNetwork:
    r"""Neural network class.
    """
    def __init__(self, layers, activation_functions):
        self.number_of_layers = len(layers) - 1
        self.biases = []
        self.weights = []

        self.bias_cache = [] # c vectors for adam
        self.bias_momentum = [] # v vectors for adam
        self.weight_cache = [] # c vectors for adam
        self.weight_momentum = [] # v vectors for adam

        self.adam_iteration = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.activation_functions = activation_functions

        if len(activation_functions) != self.number_of_layers:
            raise ValueError("Number of activation functions should match the number of layers.")

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size, x=np.sqrt(6 / (layers[0] + layers[-1]))), expr=f"$b^{{{i}}}$")
                )
                self.bias_cache.append(np.zeros(shape=size, dtype=np.float64))
                self.bias_momentum.append(np.zeros(shape=size, dtype=np.float64))
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(weight_init_function(layer_size1=size, layer_size2=layers[i+1], x=np.sqrt(6 / (layers[0] + layers[-1]))), expr=f"$(W^T)^{{{i}}}$")
                )
                self.weight_cache.append(np.zeros(shape=(size,layers[i+1]), dtype=np.float64))
                self.weight_momentum.append(np.zeros(shape=(size,layers[i+1]), dtype=np.float64))

    def __call__(self, x):
        for (weight, bias, activation_function) in zip(self.weights, self.biases, self.activation_functions):
            x = activation_function(x @ weight + bias)
        return x

    def reset_gradients(self):
        for weight in self.weights:
            weight.reset_grad()
        for bias in self.biases:
            bias.reset_grad()
    
    def reset_adam(self):
        for i, weight in enumerate(self.weights):
            self.weight_cache[i] = np.zeros_like(weight)
            self.weight_momentum[i] = np.zeros_like(weight)
        for i, bias in enumerate(self.biases):
            self.bias_cache[i] = np.zeros_like(bias)
            self.bias_momentum[i] = np.zeros_like(bias)
        self.adam_iteration = 1

    def gradient_descent(self, learning_rate):
        for weight in self.weights:
            weight.data -= learning_rate * weight.grad
        for bias in self.biases:
            bias.data -= learning_rate * bias.grad

    def adam(self,learning_rate):
        EPSILON = 1e-6
        for i, weight in enumerate(self.weights):
            self.weight_momentum[i] = self.beta1*self.weight_momentum[i] + (1-self.beta1) * weight.grad
            v_bar = self.weight_momentum[i] / (1 - self.beta1 ** self.adam_iteration)
            self.weight_cache[i] = self.beta2*self.weight_cache[i] + (1-self.beta2) * weight.grad ** 2
            c_bar = self.weight_cache[i] / (1 - self.beta2 ** self.adam_iteration)
            weight.data -= learning_rate / (EPSILON + c_bar) * v_bar
        for i, bias in enumerate(self.biases):
            self.bias_momentum[i] = self.beta1*self.bias_momentum[i] + (1-self.beta1) * bias.grad
            v_bar = self.bias_momentum[i] / (1 - self.beta1 ** self.adam_iteration)
            self.bias_cache[i] = self.beta2*self.bias_cache[i] + (1-self.beta2) * bias.grad ** 2
            c_bar = self.bias_cache[i] / (1 - self.beta2 ** self.adam_iteration)
            bias.data -= learning_rate / (EPSILON + c_bar) * v_bar
        self.adam_iteration += 1

    def save(self, path):
        np_weights = [weight.data for weight in self.weights]
        np_biases = [bias.data for bias in self.biases]

        np.savez(path / "weights.npz", *np_weights)
        np.savez(path / "biases.npz", *np_biases)

    def load(self, path):
        np_weights = [arr for arr in np.load(path / "weights.npz").values()]
        np_biases = [arr for arr in np.load(path / "biases.npz").values()]

        self.weights = [
            Value(weight, expr=f"$W^{{{i}}}$") for i, weight in enumerate(np_weights)
        ]
        self.biases = [
            Value(bias, expr=f"$b^{{{i}}}$") for i, bias in enumerate(np_biases)
        ]
