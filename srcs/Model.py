from Layer import Layer
from Optimizer import Optimizer
from Loss import CrossEntropyLoss

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


class Model():
    def __init__(self, sizes = [10, 15, 16, 2], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = Optimizer(learning_rate = 0.1, Loss = CrossEntropyLoss())):
        self.layers = make_layer_list_from_sizes_and_activations(sizes, activations)
        self.Optimizer = optimizer


    def feed_forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def fit(self, x, y):
        y_hat = self.feed_forward(x)
        self.Optimizer.fit(self.layers, y)

    def __str__(self):
        out = ""
        for l in self.layers:
            out = out + str(l)
            out = out + "   "
        return (out)
        

