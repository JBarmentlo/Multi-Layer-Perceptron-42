import numpy as np
from scipy.special import softmax, expit


def sigmoid(z):
    # a = 1/(1 + np.exp(-z))
    return (expit(z))


def sigmoid_derivative(z, a):
    return (a) * (1 - a)


def softmax_col(z):
    e = np.exp(z - np.max(z))
    s = np.sum(e, axis = 0, keepdims=True)
    return (e/s)

def softmax_row(z):
    e = np.exp(z - np.max(z))
    s = np.sum(e, axis = 1, keepdims=True)
    return (e/s)

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
        print("You have entered an incorrect activation function name, defaulting to softmax")
        return softmax_row, softmax_row_derivative
