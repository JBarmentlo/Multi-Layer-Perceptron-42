import numpy as np
from math import sqrt
from numpy.random import rand


def add_bias_units(a):
    '''
        inserts a column of ones to the left of a
    '''
    rows = a.shape[0]
    return np.concatenate((np.ones([rows, 1]), a), axis = 1)


def xavier_init(in_size, out_size):
    invsqrtn = 1.0 / sqrt(in_size)
    lower = -invsqrtn
    upper = invsqrtn
    weights = rand(in_size + 1, out_size)
    # biases = rand(out_size, 1)
    weights = weights * (upper - lower) + lower
    # biases = biases * (upper - lower) + lower
    # return (weights, biases)
    return (weights)