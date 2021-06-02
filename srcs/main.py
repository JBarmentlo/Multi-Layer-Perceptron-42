from modules import Model, Dataset, NAGOptimizer, Optimizer, KFoldIterator, CrossEntropyLoss, MSELoss
from utils import create_dataset_from_path, calculate_and_display_metrics, evaluate_binary_classifier
import numpy as np
import os

# CONFIG ARGUMENTS
folds = 5 # must be smaller than 5
reset_between_folds = False # set Trueto reset the model weights at every fold to evaluate the learning preocesstpt
epochs = 10
batchsize = 32
dataset_path = os.path.join(os.environ["BASE_DIR"], "data/data.csv")
print_at_every_epoch = True
loss = CrossEntropyLoss()
optimizer = NAGOptimizer(learning_rate = 0.03, momentum = 0.9)


def epoch_print(m, train_dataset, test_dataset):
    if (print_at_every_epoch):
        print(f"Fold: {f+ 1}/{folds}  \tEpoch: {e:4}/{epochs}   \tLoss: {m.Optimizer.Loss.loss(m.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y):.4f}")


def end_epoch_print(m, train_dataset, test_dataset):
    if (not print_at_every_epoch):
        print(f"Fold: {f+ 1}/{folds}  \tEpoch: {e:4}/{epochs}   \tLoss: {m.Optimizer.Loss.loss(m.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {m.Optimizer.Loss.loss(m.feed_forward(test_dataset.x), test_dataset.y):.4f}")


# def early_stopper(last_loss,loss)


if __name__ == "__main__":
    np.random.seed(121)
    d = create_dataset_from_path()
    kfold_iterator = KFoldIterator(d.x, d.y, 5)
    m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = optimizer)
    for f in range(folds):
        if (reset_between_folds):
            m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer = optimizer)
        try:
            train_dataset, test_dataset = next(kfold_iterator)
            for e in range(epochs):
                m.train(train_dataset, batchsize = batchsize)
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