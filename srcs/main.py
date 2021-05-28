from modules import Model, Dataset, NAGOptimizer, Optimizer, KFoldIterator
from utils import activations, create_dataset_from_path, evaluate_binary_classifier, evaluate_nonbinary_classifier
import numpy as np
import os

if __name__ == "__main__":
    np.random.seed(121)
    dataset_path = os.path.join(os.environ["BASE_DIR"], "datasets/dataset_train.csv")
    d = create_dataset_from_path()
    m = Model(sizes = [d.x.shape[1], 15, 8, d.y.shape[1]], activations = ["sigmoid", "sigmoid", "softmax"], optimizer=Optimizer())
    kfold_iterator = KFoldIterator(d.x, d.y, 5)
    folds = 1
    for f in range(folds):
        try:
            train_dataset, test_dataset = next(kfold_iterator)
            for e in range(epochs):
                for x, y in train_dataset.batchiterator(batchsize):
                    self.fit(x, y)
                print(f"Fold: {f+ 1}/{folds}  \tEpoch: {e:4}/{epochs}   \tLoss: {self.Optimizer.Loss.loss(self.feed_forward(train_dataset.x), train_dataset.y):.4f}    \tValidation Loss: {self.Optimizer.Loss.loss(self.feed_forward(test_dataset.x), test_dataset.y):.4f}")
                loss = self.Optimizer.Loss.loss(self.feed_forward(train_dataset.x), train_dataset.y)
            print("\n")
        except StopIteration:
            pass

    m.train(d, batchsize = 32, epochs=50, folds=1)
    tp, fp, tn, fn = evaluate_binary_classifier(m, d.x, d.y)
    met = evaluate_nonbinary_classifier(m, d.x, d.y)
    # a = m.feed_forward(np.ones([10, 1]))
    # kfold = d.k_fold_iter(5)
    # for xtr, ytr, xte, yte in kfold:
    #     print(xtr.shape, ytr.shape, xte.shape, yte.shape)