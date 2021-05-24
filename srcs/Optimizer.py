import numpy as np
from Loss import CrossEntropyLoss

class Optimizer():
    def __init__(self, learning_rate = 0.1, Loss = CrossEntropyLoss()):
        self.Loss = Loss
        self.lr = learning_rate
        self.local_gradient = 0


    def update_weights(self, gradient, layer):
        # print(gradient.shape)
        layer.w = layer.w - self.lr * gradient


    def compute_local_gradients(self, layers):
        pass


    def fit(self, layers, y):
        self.local_gradient = self.Loss.loss_derivative(layers[-1].a, y)
        # print("loss dev:", self.local_gradient)
        for l in reversed(layers):
            self.local_gradient, weights_gradient = l.backwards(self.local_gradient)
            # print(f"{self.local_gradient =}")
            self.update_weights(weights_gradient, l)
