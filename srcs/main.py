from modules import Model, Dataset
from utils import activations, create_dataset_from_path
import numpy as np
import os

if __name__ == "__main__":
    dataset_path = os.path.join(os.environ["BASE_DIR"], "datasets/dataset_train.csv")
    d = create_dataset_from_path(dataset_path)
    m = Model(sizes = [7, 10, 4], activations = ["sigmoid", "softmax"])
    # print(d.y[0])
    m.train(d, batchsize = 32)
    # a = m.feed_forward(np.ones([10, 1]))
    # kfold = d.k_fold_iter(5)
    # for xtr, ytr, xte, yte in kfold:
    #     print(xtr.shape, ytr.shape, xte.shape, yte.shape)
