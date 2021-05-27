import numpy as np
from .Dataset import Dataset

class KFoldIterator():
    def __init__(self, x, y, k):
        self.k = k
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.current_k = -1
        self.gen_random_idx()


    def gen_random_idx(self):
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)


    def gen_ith_fold(self, i):
        '''
            i can take values from 0 to k - 1 (included)
        '''
        test_mask = np.ones(self.size, dtype=bool)
        test_idx = np.arange(int(i * self.size / self.k), int((i + 1) * self.size / self.k), 1)
        test_idx = self.idx[test_idx]
        test_mask[test_idx] = False
        train_mask = ~test_mask
        self.x_train = self.x[train_mask, :]
        self.y_train = self.y[train_mask, :]
        self.x_test = self.x[test_mask, :]
        self.y_test = self.y[test_mask, :]


    def batchiterator(self, batchsize):
        return BatchIterator(self.x_train, self.y_train, batchsize)


    def __iter__(self):
        self.current_k = -1
        return (self)


    def __next__(self):
        '''
            # returns a (x_train, y_train, x_test, y_test) tuple
        '''
        self.current_k += 1
        if (self.current_k >= self.k):
            raise StopIteration
        else:
            self.gen_ith_fold(self.current_k)
            return Dataset(self.x_train, self.y_train, standardize = False), Dataset(self.x_test, self.y_test, standardize = False)


    
