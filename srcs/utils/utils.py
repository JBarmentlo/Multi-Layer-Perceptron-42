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


def create_dataset_from_path(path = path, standardize = True, usecols = usefull_cols, y_col = 1):
    df = pd.read_csv(path, usecols=usefull_cols, header=None)
    df.fillna(df.median(), inplace = True)
    y_df = pd.get_dummies(df[y_col] ,drop_first = False)
    df.drop([y_col], axis = 1, inplace = True)
    y = y_df.to_numpy()
    # y_p = self.y.shape[1]
    x = df.to_numpy()
    # p = self.x.shape[1]
    # m = self.x.shape[0]
    return Dataset(x, y, standardize = standardize)


def evaluate_binary_classifier(model, x, y):
    yhat = model.feed_forward(x)
    yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    e = (2 * y )+ yhatmax
    tp = (e[:, 1] == 3).astype(int).sum()
    tn = (e[:, 1] == 0).astype(int).sum()
    fn = (e[:, 1] == 2).astype(int).sum()
    fp = (e[:, 1] == 1).astype(int).sum()
    return tp, fp, tn, fn


def evaluate_nonbinary_classifier(model, x, y):
    yhat = model.feed_forward(x)
    yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    metrics = []
    for col in range(y.shape[1]):
        yy = y[:, col]
        yyhatmax = yhatmax[:, col]
        e = 2 * yy + yyhatmax
        tp = (e == 3).astype(int).sum()
        tn = (e == 0).astype(int).sum()
        fn = (e == 2).astype(int).sum()
        fp = (e == 1).astype(int).sum()
        metrics.append((tp, fp, tn, fn))
    return metrics


def calculate_metrics(tp, fp, tn, fn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2.0 * (sensitivity * precision) / (sensitivity + precision)
    return (sensitivity, specificity, precision, f1)


def print_metrics(tp, fp, tn, fn):
    sensitivity, specificity, precision, f1 = calculate_metrics(tp, fp, tn, fn)
    print(f"{sensitivity = :.2f}, {specificity = :.2f}, {precision = :.2f}, {f1 = :.2f}\n")

def calculate_and_display_metrics(model, x, y):
    '''
        Specific to a binary classifier
    '''
    tp, fp, tn, fn = evaluate_binary_classifier(model, x, y)
    print_metrics(tp, fp, tn, fn)