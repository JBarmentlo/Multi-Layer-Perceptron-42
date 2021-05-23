import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler 

categorical_cols = ["Hogwarts House", "Best Hand"]
categorical_col_prefix = ["House", "Hand"]
useless_cols = ["First Name", "Last Name", "Index"]
usefull_cols = ["Hogwarts House", "Muggle Studies", "Transfiguration", "Divination", 'Ancient Runes', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts']
path = "datasets/dataset_train.csv"

def drop_useless_cols(df, useless_cols = useless_cols):
    df.drop(useless_cols, axis = 1, inplace = True)

def format_birthdays(df):
    df["Birth Year"] = df["Birthday"].apply(lambda x: x.year).astype(float)
    df["Birth Month"] = df["Birthday"].apply(lambda x: x.month).astype(float)
    df.drop(["Birthday"], axis=1, inplace = True)


class Dataset():
    def __init__(self, path, standardize = True, test_set=False):
        self.data = None
        self.i = -1
        try:
            self.read_csv(path, test_set)
        except Exception as e:
            print(f"Please give a valid input, only numeric data is accepted\n{e}")
            raise ValueError
        self.standardized = False
        if (standardize):
            self.standardize()
        self.add_ones_to_x()


    def destandardize(self):
        if (not self.standardized):
            return
        self.x[:,1:] = self.x_scaler.inverse_transform(self.x[:,1:])


    def standardize(self):
        self.standardized = True
        self.x_scaler = StandardScaler()
        self.x_scaler.fit(self.x)
        self.x = self.x_scaler.transform(self.x)


    def read_csv(self, path, test_set):
        if (not test_set):
            self.data = pd.read_csv(path, usecols=usefull_cols).dropna().to_numpy()
            df = pd.read_csv(path, usecols=usefull_cols)
            df.fillna(df.median(), inplace = True)
            y_df = pd.get_dummies(df["Hogwarts House"] ,drop_first = False)
            df.drop(["Hogwarts House"], axis = 1, inplace = True)
            self.y = y_df.to_numpy()
            self.y_p = self.y.shape[1]
            self.x = df.to_numpy()
            self.p = self.x.shape[1]
            self.m = self.x.shape[0]
        else:
            cols = usefull_cols
            cols.remove("Hogwarts House")
            df = pd.read_csv(path, usecols=cols)
            df.fillna(df.median(), inplace = True)
            self.x = df.to_numpy()
            self.p = self.x.shape[1]
            self.m = self.x.shape[0]




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

    