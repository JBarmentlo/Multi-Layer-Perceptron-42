# import numpy as np
from math import sqrt
from numpy.random import rand


def xavier_init(in_size, out_size):
    invsqrtn = 1.0 / sqrt(in_size)
    lower = -invsqrtn
    upper = invsqrtn
    weights = rand(out_size, in_size)
    biases = rand(out_size, 1)
    weights = weights * (upper - lower) + lower
    biases = biases * (upper - lower) + lower
    return (weights, biases)