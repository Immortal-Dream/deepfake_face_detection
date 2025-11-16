import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score


class F1MetricsCallback(Callback):
    """
    Callback to compute F1 score for training and validation after each epoch.
    """

    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        # Calculate F1 on training set
        y_train_pred = self.model.predict(self.train_data[0], verbose=0)
        y_train_pred_labels = np.argmax(y_train_pred, axis=1)
        y_train_true_labels = np.argmax(self.train_data[1], axis=1)
        train_f1 = f1_score(y_train_true_labels, y_train_pred_labels, average='weighted')
        self.train_f1_scores.append(train_f1)

        # Calculate F1 on validation set
        y_val_pred = self.model.predict(self.val_data[0], verbose=0)
        y_val_pred_labels = np.argmax(y_val_pred, axis=1)
        y_val_true_labels = np.argmax(self.val_data[1], axis=1)
        val_f1 = f1_score(y_val_true_labels, y_val_pred_labels, average='weighted')
        self.val_f1_scores.append(val_f1)

        print(f" - train_f1: {train_f1:.4f} - val_f1: {val_f1:.4f}")
