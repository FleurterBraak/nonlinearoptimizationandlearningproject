import numpy as np


class Value:
    def __init__(self, data, expr: str = "", children=()):
        self.expr = expr
        if isinstance(data, Value):
            self.data = data.data
        else:
            self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self.children = set(children)

    def _backward_gradient_step(self):
        pass

    def backward(self, primary=True, direction=None):
        if primary:
            if direction is not None:
                self.grad = direction
            else:
                self.grad = np.ones_like(self.grad)

        self._backward_gradient_step()

        for child in self.children:
            child.backward(primary=False)

    def __add__(self, other):
        """This function is called when you add two Value instances with each other. The arguments `self` and
        `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data + other.data, f"({self.expr}+{other.expr})", (self, other))

        def _backward_gradient_step():
            if self.data.ndim == 2 and other.data.ndim == 1:  # matrix + vector
                self.grad += result.grad
                other.grad += np.sum(result.grad, axis=0)
            elif self.data.ndim == 1 and other.data.ndim == 2:  # vector + matrix
                self.grad += np.sum(result.grad, axis=0)
                other.grad += result.grad
            else:  # 2 matrices or 2 vectors
                self.grad += result.grad
                other.grad += result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def __mul__(self, other):
        """This function is called when you multiply two Value instances with each other. The arguments `self` and
        `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data * other.data, f"{self.expr}*{other.expr}", (self, other))

        def _backward_gradient_step():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def __matmul__(self, other):
        """This function is called when you matrix multiply two Value instances with each other. The arguments `self`
        and `other` will be the left Value instance and the right Value instance respectfully.
        """
        result = Value(self.data @ other.data, f"{self.expr}@{other.expr}", (self, other))

        def _backward_gradient_step():
            if result.data.ndim == 0:  # 2 vectors ==> scalar
                self.grad += result.grad * other.data
                other.grad += self.data * result.grad
            elif self.data.ndim == 2 and other.data.ndim == 1:  # matrix @ vector ==> vector
                self.grad += np.outer(result.grad, other.data)
                other.grad += self.data.T @ result.grad
            elif self.data.ndim == 1 and other.data.ndim == 2:  # vector @ matrix ==> vector
                self.grad += result.grad @ other.data.T
                other.grad += np.outer(self.data, result.grad)
            else:  # 2 matrices ==> matrix
                self.grad += result.grad @ other.data.T
                other.grad += self.data.T @ result.grad

        result._backward_gradient_step = _backward_gradient_step
        return result

    def reset_grad(self):
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        for child in self.children:
            child.reset_grad()

    def __repr__(self):
        return f"Value({self.data})"

    def __getitem__(self, key):
        return self.data[key]


def allclose(x1: Value | np.ndarray, x2: Value | np.ndarray, *args, **kwargs) -> bool:
    return np.allclose(
        x1.data if isinstance(x1, Value) else x1,
        x2.data if isinstance(x2, Value) else x2,
        *args,
        **kwargs
    )


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
