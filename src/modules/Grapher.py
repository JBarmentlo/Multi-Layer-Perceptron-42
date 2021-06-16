import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import evaluate_binary_classifier, calculate_metrics
from modules import Model, Dataset, Optimizer

class Grapher():
    def __init__(self):
        self.metrics = pd.DataFrame(columns=["Epoch", "Loss", "Validation Loss", "F1"])
        sns.set_theme(style="darkgrid")
        self.epoch = 0


    def add_data_point(self, loss, val_loss, f1):
        new_row = {'Epoch': self.epoch, 'Loss': loss , 'Validation Loss':val_loss, 'F1':f1}
        #append row to the dataframe
        self.metrics = self.metrics.append(new_row, ignore_index=True)

    def plot_metrics(self):
        self.metrics.Epoch = self.metrics.index
        sns.lineplot(x = "Epoch", y = "value", data = pd.melt(self.metrics, id_vars= ["Epoch"]), hue = "variable", legend="full")
        plt.show()

    def calculate_metrics(self, m : Model, train_d : Dataset, test_d : Dataset, d : Dataset) -> None:
        tp, fp, tn, fn = evaluate_binary_classifier(m, d.x, d.y)
        sensitivity, specificity, precision, f1 = calculate_metrics(tp, fp, tn, fn)
        loss = m.Optimizer.Loss.loss(m.feed_forward(train_d.x), train_d.y)
        val_loss = m.Optimizer.Loss.loss(m.feed_forward(test_d.x), test_d.y)
        self.add_data_point(loss, val_loss, f1)
        self.loss = loss
        self.val_loss = val_loss
        self.epoch += 1

    def print_metrics(self):
        print(f"Epoch: {self.epoch + 1:4}   \tLoss: {self.loss:.4f}    \tValidation Loss: {self.val_loss:.4f}")
        

