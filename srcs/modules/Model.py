from .Layer import Layer
from .Optimizer import Optimizer
from .Loss import CrossEntropyLoss
from .K_fold_iterator import KFoldIterator
from utils import delete_dir_and_contents
import os
import pickle
import numpy as np


def make_layer_list_from_sizes_and_activations(sizes, activations):
    if (len(sizes) != (len(activations) + 1)):
        print("Please enter an ctivation function for every layer")
        raise ValueError
    input_size = sizes[0]
    sizes = sizes[1:]
    layers = []
    for size, activation in zip(sizes, activations):
        layers.append(Layer(input_size, size, activation))
        input_size = size
    return layers


def load_model(model_name = "mymodel"):
    try:
        models = os.path.join(os.environ['BASE_DIR'], "models")
        path = os.path.join(models, model_name)
        activations = np.genfromtxt(os.path.join(path, "activations.csv"), delimiter=",", dtype = str)
        sizes = np.genfromtxt(os.path.join(path, "sizes.csv"), delimiter = ",", dtype = int)
        with open(os.path.join(path, "optimizer.pkl"), "rb") as f:
            Optimizer = pickle.load(f)
        m = Model(sizes, activations, optimizer=Optimizer)
        for i in range(len(sizes) - 1):
            m.layers[i].w = np.genfromtxt(os.path.join(path, f"weights_{i}.csv"), delimiter = ",")
    except:
        print("Enter a valid path please")
        raise ValueError
    return m


class Model():
    def __init__(self, sizes, activations, optimizer = Optimizer(learning_rate = 0.1, Loss = CrossEntropyLoss())):
        self.layers = make_layer_list_from_sizes_and_activations(sizes, activations)
        self.Optimizer = optimizer
        self.Optimizer.layers = self.layers
        self.sizes = sizes
        self.activations = activations


    def feed_forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def fit(self, x, y):
        self.Optimizer.pre_fit(x, y)
        y_hat = self.feed_forward(x)
        self.Optimizer.fit(self.layers, y)
        self.Optimizer.post_fit(x, y)


    def epoch(self, batchiterator):
        for x, y in batchiterator:
            self.fit(x, y)


    def train(self, dataset, batchsize = 0):
        for x, y in dataset.batchiterator(batchsize):
            self.fit(x, y)


    def save(self, model_name = "mymodel"):
        models = os.path.join(os.environ['BASE_DIR'], "models")
        path = os.path.join(models, model_name)
        delete_dir_and_contents(path)
        os.mkdir(path)
        np.savetxt(os.path.join(path, "activations.csv"), self.activations, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(path, "sizes.csv"), self.sizes, delimiter = ",", fmt="%d")
        with open(os.path.join(path, "optimizer.pkl"), "wb+") as f:
            pickle.dump(self.Optimizer, f)
        for i in range(len(self.layers)):
            np.savetxt(os.path.join(path, f"weights_{i}.csv"), self.layers[i].w, delimiter = ",")




    def __str__(self):
        out = ""
        for l in self.layers:
            out = out + str(l)
            out = out + "   "
        return (out)
        

