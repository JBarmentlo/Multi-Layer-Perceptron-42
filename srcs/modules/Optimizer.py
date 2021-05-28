import numpy as np
from .Loss import CrossEntropyLoss
import logging
optimizerlogger = logging.getLogger("Optimizer")
optimizerlogger.setLevel(logging.WARNING)

class Optimizer():
    def __init__(self, learning_rate = 0.1, Loss = CrossEntropyLoss()):
        '''
            Method : "classic", "NAG" : Nesterov Accellerated Gradient
        '''
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

    def pre_fit(self, x, y):
        pass

    
    def post_fit(self, x, y):
        pass
    


class NAGOptimizer():
    def __init__(self, learning_rate = 0.03, momentum = 0.9, Loss = CrossEntropyLoss()):
        '''
            Method : "classic", "NAG" : Nesterov Accellerated Gradient
        '''
        self.Loss = Loss
        self.lr = learning_rate
        self.last_grad = None
        self.local_gradient = 0
        self.momentum = momentum
        self.velocity = None


    def update_weights(self, gradient, layer):
        layer.w = layer.w - self.lr * gradient
        optimizerlogger.debug(f"l.w :\n{layer.w}")


    def apply_momentum_to_weights(self, layers):
        for l, v in zip(reversed(layers), self.velocity):
            l.w = l.w + self.momentum * v


    def compute_local_gradients(self, layers):
        pass


    def fit(self, layers, y):
        self.local_gradient = self.Loss.loss_derivative(layers[-1].a, y)
        optimizerlogger.debug(f"loss dev: {self.local_gradient}")
        for l, i in zip(reversed(layers), range(len(layers))):
            self.local_gradient, weights_gradient = l.backwards(djda=self.local_gradient)
            optimizerlogger.debug(f"{self.local_gradient =}")
            self.update_weights(weights_gradient, l)
            self.velocity[i] = self.momentum * self.velocity[i] - weights_gradient * self.lr
        
            

    def pre_fit(self, x, y):
        if (self.velocity == None):
            self.velocity = [0] * len(self.layers)
        self.apply_momentum_to_weights(self.layers)

    
    def post_fit(self, x, y):
        pass
    