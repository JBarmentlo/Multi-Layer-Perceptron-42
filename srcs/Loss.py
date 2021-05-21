import numpy as np

class BinaryCrossEntropyLoss():
    def __init__(self):
        pass

    def loss(self, y_hat, y):
        loss = np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return (loss)

    
    def loss_derivative(self, y_hat, y):
        dlossonda = (y_hat - y) / (y_hat * (1 - y))
        return dlossonda