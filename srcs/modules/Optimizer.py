import numpy as np
from .Loss import CrossEntropyLoss
import logging
optimizerlogger = logging.getLogger("Optimizer")
optimizerlogger.setLevel(logging.WARNING)

class Optimizer():
    def __init__(self, learning_rate = 0.1, Loss = CrossEntropyLoss(), method = "classic"):
        self.Loss = Loss
        self.lr = learning_rate
        self.last_grad = None
        self.local_gradient = 0


    def update_weights(self, gradient, layer):
        layer.w = layer.w - self.lr * gradient
        optimizerlogger.debug(f"l.w :\n{layer.w}")


    def compute_local_gradients(self, layers):
        pass


    def fit(self, layers, y):
        self.local_gradient = self.Loss.loss_derivative(layers[-1].a, y)
        optimizerlogger.debug(f"loss dev: {self.local_gradient}")
        for l in reversed(layers):
            self.local_gradient, weights_gradient = l.backwards(djda=self.local_gradient)
            optimizerlogger.debug(f"{self.local_gradient =}")
            self.update_weights(weights_gradient, l)


