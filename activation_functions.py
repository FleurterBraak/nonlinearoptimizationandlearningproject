import numpy as np

from supplementary import Value


def identity(x: Value) -> Value:
    result = Value(x.data, f"{x.expr}", (x,))

    def _backward_gradient_step():
        x.grad += result.grad  # derivative of identity is "1"

    result._backward_gradient_step = _backward_gradient_step
    return result


def relu(x: Value) -> Value:
    result = Value(np.maximum(x.data, 0), f"ReLU({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += np.heaviside(x.data, 1) * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def logi(x: Value) -> Value:
    data = 1 / (1+np.exp(-x.data))
    result = Value(data, f"logi({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += data * (1 - data) * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def softmax(x: Value) -> Value:
    exp_x = np.exp(x.data)
    if x.data.ndim == 2:
        data = exp_x / (np.sum(exp_x, keepdims=True, axis=1) + 0.000001)
    else:
        data = exp_x / np.sum(exp_x)
    result = Value(data, f"softmax({x.expr})", (x,))

    def _backward_gradient_step():
        m, n = data.shape

        outer = np.einsum('...j,...k->...jk', data, data)
        diag = np.einsum('...j,jk->...jk', data, np.eye(n))

        x.grad += np.einsum('...jk,...k->...j', diag-outer, result.grad)

    result._backward_gradient_step = _backward_gradient_step
    return result
