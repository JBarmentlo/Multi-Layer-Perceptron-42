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

def softmax_col_derivative(z, a):
    '''
        Here we will make the jacobian matrix of da/dz
        J[i, j] = da[i] / dz[j] = 
        {
            for i != j : -a[i] * a[j]
            for i == j : a[i] * (1 - a[i]) = a[i] - a[i] ** 2
        }
        there will be an extra dimension as batch dimension (it will be the first dimension)(as there are multiple examples in a)
    '''
    a = a.T
    m, n = a.shape # m = nb examples, n = nb features
    t1 = np.einsum('ik,jk->ijk') 
    # (t1 tize: n, m, n) the first dimension is of the examples (t1[0] will be the jacobian matrix for the first example)
    diag = np.einsum('ijk, ->', a, np.eye(m, m))


    # First we create for each example feature vector, it's outer product with itself
    # ( p1^2  p1*p2  p1*p3 .... )
    # # ( p2*p1 p2^2   p2*p3 .... )
    # # ( ...     )
    # tensor1 = np.einsum('ij,ik->ijk', a, a)
    # # now we create the diagonal vector
    # # ( p1  0  0  ...  )
    # # ( 0   p2 0  ...  )
    # # ( ...            )
    # tensor2 = np.einsum('ij,jk->ijk', a, np.eye(n, n))
    # # now we make
    # # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
    # # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
    # # ( ...  
    dSoftmax = tensor2 - tensor1
    return (dSoftmax)
    

def get_activation_function(activation):
        if activation == 'sigmoid':
            return sigmoid, sigmoid_derivative
        if activation == 'softmax':
            return softmax_col, softmax_col_derivative
        print("You have entered an incorrect activation function name, defaulting to softmax")
        return softmax_col, softmax_col_derivative
