from datetime import datetime
from numpy import matmul, transpose
import numpy as np
from activations import get_activation_function
from weights_init import xavier_init
from utils import add_bias_units
# float_formatter = "{:.2E}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})
float_formatter = "{:5.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# TODO: Change docstrings to represent bias units addition
class Layer():
    '''
        The layer class takes an iput of shape (in_size, b_dim) and outputs a (out_size, b_dim) output.
        d_dim is the "batch dimension" it can take any value >= 1. It represents the number of examples fed into the net.
        x is features as colums

        The weights are of size: 
        w(out_size, in_size + 1)

        Outputs:
        z = w * x
        a = activation(h)

        Representation:
        input: features as columns
        weights: weights[i] the weights for output[i], weights[:, 0] the biases
    '''
    def __init__(self, in_size, out_size, activation = 'sigmoid'):
        # * activation = 'sigmoid' or 'softmax'
        self.activation, self.activation_derivative = get_activation_function(activation)
        self.in_size = in_size
        self.out_size = out_size
        self.w = xavier_init(in_size, out_size)
        self.local_gradient = None

    
    def forward(self, x):
        '''
            Takes input x, adds bias units and computes :
            self.z = self.w * x
            self.a = self.activation(self.h)
        '''
        self.x = add_bias_units(x)
        self.z = matmul(self.w, self.x)
        self.a = self.activation(self.z)
        return (self.a)


    def backwards(self, djonda):
        '''
            Takes an local gradient vector and computes the partial derivatives in regards to weights and biases.
            returns a matrix M so that M[i, j] is the derivative of the cost J by the weight[i, j]

            Inputs:
                djonda is the derivative of the final cost J in regards to a the activated output of out layer (dj/da)

            Variable names:
                djonda = dj on da = dj/da
                djondz = dj on dz = dj/dz
                djondw = dj/dw

            returns
        '''
        # print("djonda", djonda.shape)
        # print("activation deriv shape", self.activation_derivative(self.z, self.a).shape)
        daondz = self.activation_derivative(self.z, self.a)
        print("daondz shape: ", daondz.shape)
        djondz = np.einsum('ijk,ik->ij', daondz, djonda)
        # print("x.T.shape", self.x.T.shape)
        # print("djondz shape", djondz.shape)
        djondw = matmul(djondz, self.x.T)
        next_djonda = matmul(self.w[:, 1:].T, djondz)
        # print("OUT")
        return next_djonda, djondw


    def __str__(self):
        return (f"w.shape: {self.w.shape}")