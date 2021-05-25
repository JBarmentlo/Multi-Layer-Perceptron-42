import numpy as np
import logging

losslogger = logging.getLogger("Loss")
losslogger.setLevel(logging.DEBUG)
epsilon = 0.000001

class CrossEntropyLoss():
    def __init__(self):
        self.epsilon = 0.000001
        pass

    def loss(self, y_hat, y):
        '''
            y_hat and y are of size (*, examples, features)
            the output J is of size (1)
        '''
        p = y
        q = y_hat
        logq = np.log(q)
        loss = np.sum(p * logq, axis = 1, keepdims=True)
        loss = np.mean(loss)
        return (loss)

    
    def loss_derivative(self, y_hat, y):
        '''
            The output is a matrix djda of the size of a
            where djda[*, i] is the derivative of the loss by x[i]
            where x is the *th example
        '''
        # djonda = -1 * (y / (y_hat + self.epsilon))
        # djonda = (-1 * y_hat / (y + self.epsilon)) + ((1 - y_hat) / (1 - y + self.epsilon))
        # logging.debug(f"y:\n{y}")
        # logging.debug(f"yhat:\n{y_hat}")
        djda = -1 * y / (y_hat + epsilon)
        # logging.debug(f"-y / yhat:\n{-1 * y / y_hat}")
        djda = djda / y.shape[0]
        losslogger.debug(f"djda:\n{djda}\n")
        return djda


class MSELoss():
    def __init__(self):
        self.epsilon = 0.000001
        pass

    def loss(self, y_hat, y):
        '''
            y_hat and y are of size (*, examples, features)
            the output J is of size (1)
        '''
        p = y
        q = y_hat
        loss = np.sum(np.square(q - p), axis = 1, keepdims=True)
        loss = np.mean(loss) / 2
        return (loss)

    
    def loss_derivative(self, y_hat, y):
        '''
            The output is a matrix djda of the size of a
            where djda[*, i] is the derivative of the loss by x[i]
            where x is the *th example
        '''
        djda = (y_hat - y)
        return djda


# class BinaryCrossEntropyLoss():
#     def __init__(self):
#         pass

#     def loss(self, y_hat, y):
#         loss = np.mean(np.square(y_hat - y))
#         return (loss)

    
#     def loss_derivative(self, y_hat, y):
#         dlossonda = np.mean((y_hat - y), axis = 1)
#         return np.reshape(dlossonda, [-1, 1])