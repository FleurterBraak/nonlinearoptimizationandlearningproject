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
        self.activation_functions = activation_functions

        if len(activation_functions) != self.number_of_layers:
            raise ValueError("Number of activation functions should match the number of layers.")

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size, x=np.sqrt(6 / (layers[0] + layers[-1]))), expr=f"$b^{{{i}}}$")
                )
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(weight_init_function(layer_size1=size, layer_size2=layers[i+1], x=np.sqrt(6 / (layers[0] + layers[-1]))), expr=f"$(W^T)^{{{i}}}$")
                )

    def __call__(self, x):
        for (weight, bias, activation_function) in zip(self.weights, self.biases, self.activation_functions):
            x = activation_function(x @ weight + bias)
        return x

    def reset_gradients(self):
        for weight in self.weights:
            weight.reset_grad()
        for bias in self.biases:
            bias.reset_grad()

    def gradient_descent(self, learning_rate):
        for weight in self.weights:
            weight.data -= learning_rate * weight.grad
        for bias in self.biases:
            bias.data -= learning_rate * bias.grad

    def adam(self, learning_rate, iteration: int, v_vectors: dict[str, list[np.ndarray]]|None = None, c_vectors: dict[str, list[np.ndarray]]|None = None, beta1 = 0.9, beta2 = 0.999):        
        #initialize v and c vectors for everything we take the gradient of. could also do this outside this function.
        # to do: give better variable names
        if v_vectors is None:
            v_vectors = {
                "weights": [np.zeros_like(weight) for weight in self.weights],
                "biases": [np.zeros_like(bias) for bias in self.biases]
            }
        if c_vectors is None:
            c_vectors = {
                "weights": [np.ones_like(weight) for weight in self.weights],
                "biases": [np.ones_like(bias) for bias in self.biases]
            }
        
        #initialize new versions of v_vectors and c_vectors to be returned.
        new_v = {
            "weights": [],
            "biases": []
        }
        new_c = {
            "weights": [],
            "biases": []
        }

        for i, weight in enumerate(self.weights):
            v = beta1 * v_vectors["weights"][i]  + (2 - beta1) * weight.grad
            new_v["weights"].append(v)
            v_ = v / (1 - beta1**iteration)
            c = beta2 * c_vectors["weights"][i] + (1 - beta2)* weight.grad ** 2
            new_c["weights"].append(c)
            c_ = c / (1 - beta2**iteration)
            weight.data -= learning_rate / (1e-8 + c_) * v_ 

        for i, bias in enumerate(self.biases):
            v = beta1 * v_vectors["biases"][i]  + (2 - beta1) * bias.grad
            new_v["biases"].append(v)
            v_ = v / (1 - beta1**iteration)
            c = beta2 * c_vectors["biases"][i] + (1 - beta2)* bias.grad ** 2
            new_c["biases"].append(c)
            c_ = c / (1 - beta2**iteration)
            bias.data -= learning_rate / (1e-6 + c_) * v_

        return new_v, new_c



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
