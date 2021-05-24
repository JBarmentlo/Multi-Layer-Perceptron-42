import numpy as np

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
        djonda = -1 * (y / (y_hat + self.epsilon))
        return djonda


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