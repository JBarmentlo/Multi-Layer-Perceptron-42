import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler 
# from .K_fold_iterator import KFoldIterator
from .BatchIterator import BatchIterator
import pickle
import os


class Dataset():
    def __init__(self, x, y, standardize = True, test_set=False):
        self.x = x
        self.y = y
        self.y_p = self.y.shape[1]
        self.p = self.x.shape[1]
        self.m = self.x.shape[0]
        self.standardized = False
        if (standardize):
            self.standardize()
        # self.add_ones_to_x()


    def destandardize(self):
        if (not self.standardized):
            return
        self.x[:,1:] = self.x_scaler.inverse_transform(self.x[:,1:])


    def standardize(self):
        self.standardized = True
        self.x_scaler = StandardScaler()
        self.x_scaler.fit(self.x)
        self.x = self.x_scaler.transform(self.x)


    def split_test_train(self, k = 5):
        '''
            if k is smaller than 2 it takes the default value 2
        '''
        if (k <= 1):
            k = 2
        population = self.x.shape[0] 
        idx = np.random.choice(population, int(population / k), replace=False)
        mask = np.ones(population, dtype=bool)
        mask[idx] = False
        notmask = ~mask
        self.x_train = self.x[mask, :]
        self.x_test = self.x[notmask, :]
        self.y_train = self.y[mask, :]
        self.y_test = self.y[notmask, :]


    # def k_fold_iterator(self, k):
    #     return (KFoldIterator(k, self.x, self.y))


    def batchiterator(self, batchsize):
        return BatchIterator(self.x, self.y, batchsize)


    def add_ones_to_x(self):
        self.x = np.concatenate((np.ones([self.m, 1], dtype = self.x.dtype), self.x), axis = 1)


    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    

    def __len__(self):
        return (self.data.shape[0])

    
    def __iter__(self):
        self.i = -1
        return (self)


    def __next__(self):
        self.i += 1
        if (self.i < len(self)):
            return self[self.i]
        else:
            self.i = -1
            raise StopIteration


    def save_norm(self, model_name):
        models = os.path.join(os.environ['BASE_DIR'], "models")
        path = os.path.join(models, model_name)
        path = os.path.join(path, "norm.pkl")
        with open(path, "wb+") as f:
            pickle.dump(self.x_scaler, f)


    def use_norm(self, model_name):
        '''
            experimental
        '''
        models = os.path.join(os.environ['BASE_DIR'], "models")
        path = os.path.join(models, model_name)
        path = os.path.join(path, "norm.pkl")
        with open(path, "rb+") as f:
            self.x_scaler = pickle.load(f)
        self.x = self.x_scaler.transform(self.x)


class DummyDataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def batchiterator(self, batchsize):
        return BatchIterator(self.x, self.y, batchsize)
