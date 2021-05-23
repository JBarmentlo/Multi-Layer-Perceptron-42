import numpy as np
from scipy.special import softmax, expit


def sigmoid(z):
    # a = 1/(1 + np.exp(-z))
    return (expit(z))


def sigmoid_derivative(z, a):
    return (a) * (1 - a)


def sofmax_row(z):
    return (softmax(z, axis = 0))


def sofmax_row_derivative(z, a):
    raise NotImplementedError

def get_activation_function(activation):
        if activation == 'sigmoid':
            return sigmoid, sigmoid_derivative
        if activation == 'softmax':
            return sofmax_row, sofmax_row_derivative
        print("You have entered an incorrect activation function name, defaulting to softmax")
        return sofmax_row


def get_activation_function_derivative():
    pass
