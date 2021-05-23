import numpy as np

class BinaryCrossEntropyLoss():
    def __init__(self):
        pass

    def loss(self, y_hat, y):
        p = y
        q = y_hat
        logq = np.log(q)
        loss = np.sum(p * logq, axis = 1, keepdims=True)
        loss = np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return (loss)

    
    def loss_derivative(self, y_hat, y):
        djonda = -1 * (y / y_hat)
        return djonda


# class BinaryCrossEntropyLoss():
#     def __init__(self):
#         pass

#     def loss(self, y_hat, y):
#         loss = np.mean(np.square(y_hat - y))
#         return (loss)

    
#     def loss_derivative(self, y_hat, y):
#         dlossonda = np.mean((y_hat - y), axis = 1)
#         return np.reshape(dlossonda, [-1, 1])