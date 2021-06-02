from modules import Model, Dataset, NAGOptimizer, Optimizer, KFoldIterator, CrossEntropyLoss, MSELoss
from utils import create_dataset_from_path, calculate_and_display_metrics, evaluate_binary_classifier, is_overfitting
import numpy as np
import os
from collections import deque
import argparse


# CONFIG ARGUMENTS
folds = 5 # must be smaller than 5
reset_between_folds = False # set Trueto reset the model weights at every fold to evaluate the learning preocesstpt
epochs = 100
batchsize = 32
dataset_path = os.path.join(os.environ["BASE_DIR"], "data/data.csv")
print_at_every_epoch = True
loss = CrossEntropyLoss()
optimizer = NAGOptimizer(learning_rate = 0.03, momentum = 0.9)

usecols = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

def epoch_print(m, train_dataset, test_dataset):
    if (print_at_every_epoch):
        print(f"Fold: {f+ 1}/{folds}  \tEpoch: {e + 1:4}/{epochs}   \tLoss: {m.Optimizer.Loss.loss(m.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y):.4f}")


def end_epoch_print(m, train_dataset, test_dataset):
    print(f"Fold: {f+ 1}/{folds}  \tLoss: {m.Optimizer.Loss.loss(m.feed_forward(d.x), d.y):.4f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='A typical logistic regression')
    parser.add_argument("data_path", nargs = '?', default="data/data.csv", help="path to input data")
    parser.add_argument("--usecols", nargs="+", type=int, help= "The indexes of the colums to keep in the dataset, must be floats, must include ycol", default=usecols)
    parser.add_argument("--ycol", action="store", nargs="?", type=int, help = "Index of the column to predict", default = 1)
    parser.add_argument("--y_not_categorical", action="store_false", help="Put this option is y is not categorical")
    return parser.parse_args()

if __name__ == "__main__":
    # np.random.seed(121)
    args = parse_arguments()
    d = create_dataset_from_path(args.data_path, usecols = args.usecols, y_col = args.ycol, y_categorical = args.y_not_categorical)
    kfold_iterator = KFoldIterator(d.x, d.y, 5)
    losses = deque(maxlen=3)
    m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = optimizer)
    for f in range(folds):
        if (reset_between_folds):
            m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = optimizer)
        try:
            train_dataset, test_dataset = next(kfold_iterator)
            for e in range(epochs):
                m.train(train_dataset, batchsize = batchsize)
                losses.append(m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y))
                if (is_overfitting(losses)):
                    break
                epoch_print(m, train_dataset, test_dataset)
            end_epoch_print(m, train_dataset, test_dataset)
            print("\n")
        except StopIteration:
            pass
    # m.train(d, batchsize = 32, epochs=50, folds=1)
    calculate_and_display_metrics(m, d.x, d.y)
    tp, fp, tn, fn = evaluate_binary_classifier(m, d.x, d.y)
    m.save()
    # a = m.feed_forward(np.ones([10, 1]))
    # kfold = d.k_fold_iter(5)
    # for xtr, ytr, xte, yte in kfold:
    #     print(xtr.shape, ytr.shape, xte.shape, yte.shape)