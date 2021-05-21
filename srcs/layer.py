
from datetime import datetime
from numpy import matmul, transpose
import numpy as np
from activations import get_activation_function
from weights_init import xavier_init

# float_formatter = "{:.2E}".format
# np.set_printoptions(formatter={'float_kind':float_formatter})
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


class Layer():
    '''
        The layer class takes an iput of shape (in_size, b_dim) and outputs a (out_size, b_dim) output.
        d_dim is the "batch dimension" it can take any value >= 1. It represents the number of examples fed into the net.
        
        The weights and biases are of size: 
        w(out_size, in_size)
        b(out_size, 1)

        Outputs:
        h = w * x + b
        z = activation(h)

        Representation:
        input: features as columns
        weights: weights[i] the weights for output[i]
    '''
    def __init__(self, in_size, out_size, activation = 'sigmoid'):
        # * activation = 'sigmoid' or 'softmax'
        self.activation = get_activation_function(activation)
        self.in_size = in_size
        self.out_size = out_size
        self.w, self.b = xavier_init(in_size, out_size)

    
    def forward(self, x):
        '''
            Takes input x and computes :
            self.h = self.w * x + self.b
            self.z = self.activation(self.h)
        '''
        self.h = matmul(self.w, x) + self.b
        self.z = self.activation(self.h)



    def time_stop(self):
        return ((datetime.now() - self.start_time).seconds >= self.time_limit)


    def predict(self, data, thetas = None):
        """
            Compute prediction for given data.

            Args:
                data (np.ndarray): data to use as input
                thetas (np.ndarray, optional): Replace the thetas with your custom ones.

            Returns:
                np.ndarray: predicted outcome y = sigmoid(data * transpose(thetas)).
        """        
        if thetas is None:
            thetas = self.thetas
        if (data.ndim == 1):
            data = data[np.newaxis, :]
        out = matmul(data, transpose(self.thetas))
        return (self.activation(out))


    def error(self, thetas = None)->np.ndarray:
        """
            Compute the error matrix for the dataset.

            Args:
                thetas (ndarray, optional): Thetas to use for prediction. Defaults to the current thetas.

            Returns:
                error (np.ndarray): the error matrix = x_pred - y
        """        
        if thetas is None:
            thetas = self.thetas
        predictions = self.predict(self.dataset.x, thetas)
        error = predictions - self.dataset.y
        return (error)


    def mean_squared_error(self, thetas = None):
        if thetas is None:
            thetas = self.thetas   
        squared_error = np.square(self.error(thetas))
        self.error()
        return (np.mean(squared_error) / 2)


    def compute_gradients(self):
        x_pred = self.predict(self.dataset.x)
        error = x_pred - self.dataset.y
        gradients = matmul(transpose(self.dataset.x), error) / len(error)
        return (transpose(gradients))


    def compute_stochastic_gradients(self, batch_size):
        datasize = self.dataset.x.shape[0]
        if (batch_size >= datasize):
            batch_size = datasize
        mask = np.random.choice(datasize, batch_size, replace=False)
        x_pred = self.predict(self.dataset.x[mask, :])
        error = x_pred - self.dataset.y[mask, :]
        gradients = matmul(transpose(self.dataset.x[mask]), error) / len(error)
        return (transpose(gradients))


    def update_thetas(self):
        self.old_thetas = self.thetas
        gradients = self.compute_stochastic_gradients(batch_size=self.batch_size)
        self.thetas = self.thetas - (self.lr * gradients)

    
    def update_costs(self):
        tmpold = self.oldcost
        self.oldcost = self.newcost
        self.newcost = self.mean_squared_error()
        return tmpold


    def undo_update_costs(self, tmpold):
        self.newcost =self.oldcost
        self.oldcost = tmpold


    def training_loop(self):
        self.start_time = datetime.now()
        self.newcost = self.mean_squared_error()
        while (not self.time_stop()):
            tmpold = self.oldcost
            self.update_thetas()
            tmpold = self.update_costs()
            if (self.newcost > self.oldcost):
                self.lr = self.lr * self.lr_decay
                self.thetas = self.old_thetas
                self.undo_update_costs(tmpold)


    def unstandardize_thetas(self):
        at = self.thetas[:, 1:]
        pt = self.thetas[:, 1]
        scaler = self.dataset.x_scaler
        at = at / scaler.scale_
        pt = pt - np.dot(at, scaler.mean_)
        self.thetas = np.concatenate((pt[:, np.newaxis], at), axis = 1)


    def write_thetas_to_file(self, filename="thetas.csv"):
        if self.standardize:
            self.unstandardize_thetas()
        np.savetxt(filename, self.thetas, delimiter = ',')


    def __str__(self):
        return (f"Cost: {self.newcost}, Thetas: {self.thetas}, LR {self.lr:4.2E}")


def add_complementary_column(a):
    b = np.ones([len(a)]) - np.sum(a, axis=1)
    full = np.concatenate((a, b[:, np.newaxis]), axis = 1)
    return (full)

def evaluate_precision(shaman, x, y):
    x_pred = shaman.predict(x)
    errors = 0
    for i in range(len(y)):
        if (np.argmax(x_pred[i, :]) != np.argmax(y[i, :])):
            errors += 1
    return (1.0 - errors / len(y))