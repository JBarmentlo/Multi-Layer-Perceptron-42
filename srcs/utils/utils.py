import numpy as np
from math import sqrt
from numpy.random import rand
from modules import Dataset
import pandas as pd


def add_bias_units(a):
    '''
        inserts a column of ones to the left of a
    '''
    rows = a.shape[0]
    return np.concatenate((np.ones([rows, 1]), a), axis = 1)


def xavier_init(in_size, out_size):
    np.random.seed(12)
    invsqrtn = 1.0 / sqrt(in_size)
    lower = -invsqrtn
    upper = invsqrtn
    weights = rand(in_size + 1, out_size)
    # biases = rand(out_size, 1)
    weights = weights * (upper - lower) + lower
    # biases = biases * (upper - lower) + lower
    # return (weights, biases)
    return (weights)


categorical_cols = ["Hogwarts House", "Best Hand"]
categorical_col_prefix = ["House", "Hand"]
useless_cols = ["First Name", "Last Name", "Index"]
usefull_cols = ["Hogwarts House", "Muggle Studies", "Transfiguration", "Divination", 'Ancient Runes', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts']
path = "datasets/dataset_train.csv"

def format_birthdays(df):
    df["Birth Year"] = df["Birthday"].apply(lambda x: x.year).astype(float)
    df["Birth Month"] = df["Birthday"].apply(lambda x: x.month).astype(float)
    df.drop(["Birthday"], axis=1, inplace = True)


def create_dataset_from_path(path, standardize = True, usecols = usefull_cols, y_col = "Hogwarts House"):
    df = pd.read_csv(path, usecols=usefull_cols)
    df.fillna(df.median(), inplace = True)
    y_df = pd.get_dummies(df[y_col] ,drop_first = False)
    df.drop(["Hogwarts House"], axis = 1, inplace = True)
    y = y_df.to_numpy()
    # y_p = self.y.shape[1]
    x = df.to_numpy()
    # p = self.x.shape[1]
    # m = self.x.shape[0]
    return Dataset(x, y, standardize = standardize)