import numpy as np

class BinaryCrossEntropyLoss():
    def __init__(self):
        pass

    def loss(self, y_hat, y):
        loss = np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return (loss)

    
    def loss_derivative(self, y_hat, y):
        dlossonda = np.mean((y_hat - y) / (y_hat * (1 - y)  + 0.000001), axis = 1)
        dlossonda = np.reshape(dlossonda, [-1, 1])
        return dlossonda


# class BinaryCrossEntropyLoss():
#     def __init__(self):
#         pass

#     def loss(self, y_hat, y):
#         loss = np.mean(np.square(y_hat - y))
#         return (loss)

    
#     def loss_derivative(self, y_hat, y):
#         dlossonda = np.mean((y_hat - y), axis = 1)
#         return np.reshape(dlossonda, [-1, 1])