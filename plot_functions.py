import numpy as np
import scipy.special as sp
import math
import matplotlib.pyplot as plt

from supplementary import Value

from activation_functions import identity, relu, logi, step, softmax, tanh, sin, silu, softsign, softplus, erf


functions = [identity, relu, logi, step, softmax, tanh, sin, silu, softsign, softplus, erf]
x_axis = Value(np.linspace(-5,5, 1000), "x-as")
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 22}


for func in functions:
    plt.figure()
    plt.rc('font', **font)
    plt.title(f"{func.__name__}")
    plt.plot(x_axis.data, func(x_axis).data)
    plt.grid()
    plt.show()
