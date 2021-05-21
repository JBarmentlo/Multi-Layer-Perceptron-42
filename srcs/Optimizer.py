import numpy as np
from Loss import BinaryCrossEntropyLoss

class Optimizer():
    def __init__(self, learning_rate = 0.1, Loss = BinaryCrossEntropyLoss()):
        self.Loss = Loss
        self.lr = learning_rate