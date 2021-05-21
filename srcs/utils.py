import numpy as np

def add_bias_units(a):
    '''
        inserts a row of ones to the top of a
    '''
    cols = a.shape[1]
    return np.concatenate((np.ones([1, cols]), a), axis = 0)