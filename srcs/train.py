from modules import Model, Dataset, NAGOptimizer, Optimizer, KFoldIterator, CrossEntropyLoss, MSELoss, Grapher
from utils import create_dataset_from_path, calculate_and_display_metrics, evaluate_binary_classifier, is_overfitting
import numpy as np
import os
from collections import deque
import argparse


# CONFIG ARGUMENTS
folds = 5
reset_between_folds = False # set Trueto reset the model weights at every fold to evaluate the learning preocesstpt
epochs = 200
batchsize = 32
loss = CrossEntropyLoss()
optimizer = NAGOptimizer(learning_rate = 0.03, momentum = 0.9)

dataset_path = os.path.join(os.environ["BASE_DIR"], "data/data.csv")
ycol = 1
y_categorical = True
usecols = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
model_name = "mymodel"


def epoch_print(m, train_dataset, test_dataset):
    print(f"Epoch: {e + 1:4}/{epochs}   \tLoss: {m.Optimizer.Loss.loss(m.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y):.4f}")


def end_epoch_print(m, d):
    print(f"Loss: {m.Optimizer.Loss.loss(m.feed_forward(d.x), d.y):.4f}")


if __name__ == "__main__":
    np.random.seed(45)
    grapher = Grapher()
    d = create_dataset_from_path(dataset_path, usecols = usecols, y_col = ycol, y_categorical = y_categorical)
    kfold_iterator = KFoldIterator(d.x, d.y, folds)
    m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = optimizer)
    train_dataset, test_dataset = next(kfold_iterator)
    losses = deque(maxlen=5)
    for e in range(epochs):
        m.train(train_dataset, batchsize = batchsize)
        loss = m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y)
        grapher.add_data_point(e , loss, 0,0)
        losses.append(loss)
        if (is_overfitting(losses)):
            break
        epoch_print(m, train_dataset, test_dataset)
    end_epoch_print(m, d)
    print("\n")
    calculate_and_display_metrics(m, d.x, d.y)
    grapher.plot_metrics()
    m.save(model_name)
    d.save_norm(model_name)
    # a = m.feed_forward(np.ones([10, 1]))
    # kfold = d.k_fold_iter(5)
    # for xtr, ytr, xte, yte in kfold:
    #     print(xtr.shape, ytr.shape, xte.shape, yte.shape)