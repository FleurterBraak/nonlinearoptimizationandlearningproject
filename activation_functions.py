import numpy as np
import scipy.special as sp
import math

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
    data = 1 / (1 + np.exp(-x.data))
    result = Value(data, f"logi({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += data * (1 - data) * result.grad

    result._backward_gradient_step = _backward_gradient_step
    return result


def softmax(x: Value) -> Value:
    #exp_x = np.exp(x.data)
    if x.data.ndim == 2:
        #data = exp_x / (np.sum(exp_x, keepdims=True, axis=1) + 0.000001)
        data = sp.softmax(x.data)
    else:
        #data = exp_x / np.sum(exp_x)
        data = sp.softmax(x.data)
    result = Value(data, f"softmax({x.expr})", (x,))

    def _backward_gradient_step():
        m, n = data.shape

        outer = np.einsum("...j,...k->...jk", data, data)
        diag = np.einsum("...j,jk->...jk", data, np.eye(n))

        x.grad += np.einsum("...jk,...k->...j", diag - outer, result.grad)

    result._backward_gradient_step = _backward_gradient_step
    return result


def tanh(x: Value) -> Value:
    data = (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
    # tanh(x) = sinh(x)/cosh(x)
    result = Value(data, f"tanh({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += (1 - data**2) * result.grad
        # d/dx tanh(x) = sech(x)^2 = 1 - tanh(x)^2

    result._backward_gradient_step = _backward_gradient_step
    return result


def sin(x: Value) -> Value:
    data = np.sin(x.data)
    result = Value(data, f"sin({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += np.cos(x.data) * result.grad
        # d/dx sin(x) = cos(x)

    result._backward_gradient_step = _backward_gradient_step
    return result


def silu(x: Value) -> Value:  # SiLU (sigmoid linear unit)
    logi_data = 1 / (1 + np.exp(-x.data))
    data = x.data * logi_data
    # silu(x) = x / (1+exp(-x)) = x * logi(x)
    result = Value(data, f"swish({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += (logi_data + x.data * logi_data * (1 - logi_data)) * result.grad
        # d/dx silu(x) = logi(x) + x * d/dx logi(x) = logi(x) + x * logi(x) * (1-logi(x))

    result._backward_gradient_step = _backward_gradient_step
    return result


def softsign(x: Value) -> Value:  # SoftSign
    data = x.data / (1 + np.abs(x.data))
    # softsign(x) = x / (1 + |x|)
    result = Value(data, f"softsign({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += 1 / (1 + np.abs(x.data)) ** 2 * result.grad
        # d/dx softsign(x) = 1/(1+|x|)^2

    result._backward_gradient_step = _backward_gradient_step
    return result


def elu(x: Value, a=1) -> Value:  # ELU (exponential linear unit)
    data = x.data * np.heaviside(x.data, 1) + a * (np.exp(x.data) - 1) * np.heaviside(-x.data, 0)
    # the above works for smaller values, but for large values of x.data it causes an overflow. 
    #if np.ndim(x.data) == 1:
    #    data = np.array([x.data[i] if x.data[i] >= 0 else a * (np.exp(x.data[i]) - 1) for i in range(len(x.data))])
    #else:
    #    m,n = np.shape(x.data)
    #    data = np.zeros_like(x.data)
    #    for i in range(m):
    #        for j in range(n):
    #            data[i][j] = x.data[i][j] if x.data[i][j] >= 0 else a * (np.exp(x.data[i][j]) - 1)

    # elu(x) = {x if x >= 0, a(exp(x)-1) if x < 0}
    result = Value(data, f"elu({x.expr})", (x,))

    def _backward_gradient_step():
        #if np.ndim(x.data) == 1:
        #    gr = np.array([1 if x.data[i] >= 0 else a * np.exp(x.data[i]) for i in range(len(x.data))])
        #else:
        #    m,n = np.shape(x.data)
        #    gr = np.zeros_like(x.data)
        #    for i in range(m):
        #        for j in range(n):
        #            data[i][j] = 1 if x.data[i][j] >= 0 else a * np.exp(x.data[i][j])
        #x.grad += gr * result.grad
        x.grad += np.heaviside(x.data, 1) + np.heaviside(-x.data, 0) * a * np.exp(x.data)
        # d/dx elu = {1 if x >= 0, a * exp(x) if x<0}

    result._backward_gradient_step = _backward_gradient_step
    return result


def softplus(x: Value) -> Value:  # SoftPlus
    data = sp.softplus(x.data)
    #data = np.log(1 + np.exp(x.data))
    # softplus(x) = ln(1 + exp(x))
    result = Value(data, f"sofplus({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += 1 / (1 + np.exp(-x.data)) * result.grad
        # d/dx softplus(x) = logi(x)

    result._backward_gradient_step = _backward_gradient_step
    return result

def erf(x: Value) -> Value:
    if x.data.ndim == 2:
        data = np.array([np.array([math.erf(t) for t in x.data[i]]) for i in range(len(x.data))])
    else:
        data = np.array([math.erf(t) for t in x.data])
    result = Value(data, f"erf({x.expr})", (x,))

    def _backward_gradient_step():
        x.grad += 2/np.sqrt(np.pi) * np.exp(-x.data**2) * result.grad
    
    result._backward_gradient_step = _backward_gradient_step
    return result