from .Layer import Layer
from .Optimizer import Optimizer
from .Loss import CrossEntropyLoss
from .K_fold_iterator import KFoldIterator

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
    def __init__(self, sizes = [7, 15, 16, 2], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = Optimizer(learning_rate = 0.1, Loss = CrossEntropyLoss())):
        self.layers = make_layer_list_from_sizes_and_activations(sizes, activations)
        self.Optimizer = optimizer
        self.Optimizer.layers = self.layers

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


    def train(self, dataset, epochs = 100, batchsize = 0, folds = 1, train_test_ratio = 5):
        kfold_iterator = KFoldIterator(dataset.x, dataset.y, train_test_ratio)
        for f in range(folds):
            try:
                train_dataset, test_dataset = next(kfold_iterator)
                # last_loss = 0
                for e in range(epochs):
                    for x, y in train_dataset.batchiterator(batchsize):
                        self.fit(x, y)
                    print(f"Fold: {f+ 1}/{folds}  \tEpoch: {e:4}/{epochs}   \tLoss: {self.Optimizer.Loss.loss(self.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {self.Optimizer.Loss.loss(self.feed_forward(test_dataset.x), test_dataset.y):.4f}")
                    loss = self.Optimizer.Loss.loss(self.feed_forward(train_dataset.x), train_dataset.y)
                    # print("loss ratio: ", last_loss / loss)
                    # last_loss = loss
                print("\n")
            except StopIteration:
                pass

    def __str__(self):
        out = ""
        for l in self.layers:
            out = out + str(l)
            out = out + "   "
        return (out)
        

