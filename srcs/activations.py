import numpy as np



def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sofmax_row(x):
    return (softmax(x, axis = 1))


def get_activation_function(activation):
        if activation == 'sigmoid':
            return sigmoid
        if activation == 'softmax':
            return sofmax_row
        print("You have entered an incorrect activation function name, defaulting to softmax")
        return sofmax_row


def get_activation_function_derivative():
    pass
