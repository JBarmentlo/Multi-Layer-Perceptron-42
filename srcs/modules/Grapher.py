import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Grapher():
    def __init__(self):
        self.metrics = pd.DataFrame(columns=["Epoch", "Loss", "Validation Loss", "F1"])
        sns.set_theme(style="darkgrid")


    def add_data_point(self, epoch, loss, val_loss, f1):
        new_row = {'Epoch': epoch, 'Loss': loss , 'Validation Loss':val_loss, 'F1':f1}
        #append row to the dataframe
        self.metrics = self.metrics.append(new_row, ignore_index=True)

    def plot_metrics(self):
        self.metrics.Epoch = self.metrics.index
        sns.lineplot(x = "Epoch", y = "F1", data = self.metrics, legend="full")
        sns.lineplot(x = "Epoch", y = "Loss", data = self.metrics, legend="full")
        plt.show()
