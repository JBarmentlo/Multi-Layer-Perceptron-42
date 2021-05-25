import numpy as np
from scipy.special import softmax, expit
import logging

activationslogger = logging.getLogger("Activations")
activationslogger.setLevel(logging.WARNING)
epsilon = 0.00001

def sigmoid(z):
    # a = 1/(1 + np.exp(-z))
    return (expit(z))


def sigmoid_derivative(z, a):
    da = (a) * (1 - a)
    b, n = da.shape
    # print("da sig\n", da)
    # print(f"{da[0] =} , \n{da[1] = }")
    da = np.einsum('ij,jk->ijk' , da, np.eye(n, n))
    # print(f"{da[0] =} , \n{da[1] = }")
    # print("da sig\n", da)
    return da


def softmax_col(z):
    e = np.exp(z - np.max(z))
    s = np.sum(e, axis = 0, keepdims=True)
    return (e/s)


def identity(z):
    return (z)


def identyty_derivative(z, a):
    b, n = a.shape
    # print("da sig\n", da)
    # print(f"{da[0] =} , \n{da[1] = }")
    da = np.einsum('ij,jk->ijk' , np.ones(a.shape), np.eye(n, n))
    return da


def softmax_row(z):
    activationslogger.debug("Softmax call")
    activationslogger.debug(f"z.shape: {z.shape}\nz:\n{z}\nnp.nax.z:\n{z.max(axis=1)}\n")
    z = z - z.max(axis = 1, keepdims=True)
    activationslogger.debug(f"z:\n{z}\n")
    e = np.exp(z)
    activationslogger.debug(f"e:\n{e}")
    s = np.sum(e, axis = 1, keepdims=True)
    activationslogger.debug(f"s : {s}")
    return (e/(s))


def softmax_row_derivative(z, a):
    '''
        Here we will make the jacobian matrix of da/dz
        with a = softmax(z)
        J[i, j] = da[i] / dz[j] = 
        {
            for i != j : -a[i] * a[j]
            for i == j : a[i] * (1 - a[i]) = a[i] - a[i] ** 2
        }
        there will be an extra dimension as batch dimension (it will be the first dimension)(as there are multiple examples in a)
    '''
    m, n = a.shape # m = nb examples, n = nb features
    t1 = np.einsum('ij,ik->ijk', a, a)
    # (t1 tize: m, n, n) the first dimension is of the examples (t1[0] will be the jacobian matrix for the first example)
    diag = np.einsum('ik,jk->ijk', a, np.eye(n, n))
    substract = diag - t1
    return substract
    

def get_activation_function(activation):
        if activation == 'sigmoid':
            return sigmoid, sigmoid_derivative
        if activation == 'softmax':
            return softmax_row, softmax_row_derivative
        if activation == "identity":
            return identity, identyty_derivative
        print("You have entered an incorrect activation function name, defaulting to softmax")
        return softmax_row, softmax_row_derivative
