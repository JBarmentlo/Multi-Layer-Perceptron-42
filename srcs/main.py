from modules import Model, Dataset
from utils import activations
import numpy as np
import os

if __name__ == "__main__":
    dataset_path = os.path.join(os.environ["BASE_DIR"], "datasets/dataset_train.csv")
    d = Dataset(dataset_path)
    m = Model(sizes = [8, 10, 4], activations = ["sigmoid", "softmax"])
    # print(d.y[0])
    d.split_test_train(5)
    for i in range(1000):
        # print(i)
        m.fit(d.x_train, d.y_train)
    out = m.feed_forward(d.x_test)
    print(np.mean(np.abs(out - d.y_test)))
    # a = m.feed_forward(np.ones([10, 1]))
    # kfold = d.k_fold_iter(5)
    # for xtr, ytr, xte, yte in kfold:
    #     print(xtr.shape, ytr.shape, xte.shape, yte.shape)
