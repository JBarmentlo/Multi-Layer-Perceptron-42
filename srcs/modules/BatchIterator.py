import numpy as np
from math import ceil


class BatchIterator():
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.size = self.x.shape[0]
        self.gen_random_idx()
        self.batch_num_max = int(ceil(self.size / self.batch_size))
        self.batch_num_current = -1
        if (self.batch_size <= 0 or self.batch_size > self.size):
            self.batch_size =self.size
        

    def gen_random_idx(self):
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)

    
    def gen_ith_batch(self, i):
        '''
            i can take values from 0 to batch_num_max - 1(included)
            returns a x,y tuple
        '''
        mask = np.zeros(self.size, dtype=bool)
        maxidx = min((i + 1) * self.batch_size, self.size)
        idx = np.arange(i * self.batch_size, max_idx, 1)
        idx = self.idx[idx]
        mask[idx] = True
        return (self.x[mask, :] , self.y[mask, :])

    def __iter__(self):
        self.current_k = -1
        return (self)

    def __next__(self):
        self.batch_num_current += 1

        if (self.batch_num_current >= self.batch_num_max):
            raise StopIteration
        else:
            return self.gen_ith_batch(self.batch_num_current)
