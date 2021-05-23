from Model import Model
from dataset import Dataset
import numpy as np

if __name__ == "__main__":
    d = Dataset("../datasets/dataset_train.csv")
    m = Model(sizes = [8, 4], activations = ["sigmoid"])
    print(d.y[0])
    for i in range(100):
        print(i)
        m.fit(d.x.T, d.y.T)
    out = m.feed_forward(d.x.T)
    # a = m.feed_forward(np.ones([10, 1]))
