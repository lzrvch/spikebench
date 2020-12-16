import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from tensorflow.keras.callbacks import Callback

from IPython.display import clear_output


class PlotLearningCurveCallback(Callback):
    def __init__(self, *args, update_freq=1, loss_log_freq=10,
                 plot_height=400, plot_width=800, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_update_freq = update_freq
        self.loss_log_freq = loss_log_freq
        self.plot_height = plot_height
        self.plot_width = plot_width

    def on_train_begin(self, logs={}):
        self.iteraton = 0
        self.iteraton_list = []
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.iteraton += 1

        if (self.iteraton % self.loss_log_freq) == 0:
            self.logs.append(logs)
            self.iteraton_list.append(self.iteraton)
            self.train_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.train_acc.append(logs.get('accuracy'))
            self.val_acc.append(logs.get('val_accuracy'))

        if (self.iteraton % self.plot_update_freq) == 0:
            clear_output(wait=True)

            loss_df = pd.DataFrame(
                {
                    'epoch': self.iteraton_list + self.iteraton_list,
                    'loss': self.train_losses + self.val_losses,
                    'dataset': ['train'] * len(self.train_losses)
                    + ['val'] * len(self.val_losses),
                }
            )
            loss_fig = px.line(
                loss_df,
                x='epoch',
                y='loss',
                title='loss',
                color='dataset',
                height=self.plot_height,
                width=self.plot_width,
            )

            acc_df = pd.DataFrame(
                {
                    'epoch': self.iteraton_list + self.iteraton_list,
                    'accuracy': self.train_acc + self.val_acc,
                    'dataset': ['train'] * len(self.train_acc)
                    + ['val'] * len(self.val_acc),
                }
            )
            acc_fig = px.line(
                acc_df,
                x='epoch',
                y='accuracy',
                title='accuracy',
                color='dataset',
                height=self.plot_height,
                width=self.plot_width,
            )
            loss_fig.show()
            acc_fig.show()
