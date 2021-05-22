from Layer import Layer
from Optimizer import Optimizer
from Loss import BinaryCrossEntropyLoss

def make_layer_list_from_sizes_and_activations(sizes, activations):
    if (len(sizes) != (len(activations) + 1)):
        print("Please enter an ctivation function for every layer")
        raise ValueError
    input_size = sizes[0]
    sizes = sizes[1:]
    print(sizes)
    layers = []
    for size, activation in zip(sizes, activations):
        layers.append(Layer(input_size, size, activation))
        input_size = size
    return layers


class Model():
    def __init__(self, sizes = [10, 15, 16, 2], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = Optimizer(learning_rate = 0.1, Loss = BinaryCrossEntropyLoss())):
        self.layers = make_layer_list_from_sizes_and_activations(sizes, activations)
        self.Optimizer = optimizer



    def feed_forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def __str__(self):
        out = ""
        for l in self.layers:
            out = out + str(l)
            out = out + "   "
        return (out)
        

import numpy as np
if __name__ == "__main__":
    m = Model()
    print(m)
    a = m.feed_forward(np.ones([10, 1]))