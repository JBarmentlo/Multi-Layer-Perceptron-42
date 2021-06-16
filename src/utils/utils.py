import numpy as np
from math import sqrt
from numpy.random import rand
from modules import Dataset
import pandas as pd
import os
import pickle

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


# usefull_cols = ["Hogwarts House", "Muggle Studies", "Transfiguration", "Divination", 'Ancient Runes', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts']
usefull_cols = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

path = "data/data.csv"

def format_birthdays(df):
    df["Birth Year"] = df["Birthday"].apply(lambda x: x.year).astype(float)
    df["Birth Month"] = df["Birthday"].apply(lambda x: x.month).astype(float)
    df.drop(["Birthday"], axis=1, inplace = True)


def create_dataset_from_path(path = path, usecols = usefull_cols, y_col = 1, y_categorical = True):
    df = pd.read_csv(path, usecols=usecols, header=None)
    df.fillna(df.median(), inplace = True)
    if (y_categorical):
        y_df = pd.get_dummies(df[y_col] ,drop_first = False)
    else:
        y_df = df[y_col]
    df.drop([y_col], axis = 1, inplace = True)
    y = y_df.to_numpy()
    x = df.to_numpy()
    return Dataset(x, y, standardize = True)


def is_overfitting(losses):
    if len(losses) < losses.maxlen:
        return False
    for i in range(losses.maxlen - 1):
        if (losses[i] > losses[i + 1]):
            return False
    return True
