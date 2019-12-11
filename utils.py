from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle
import numpy as np


def train_test_split(x_data, y_data, train_ratio):
    num_train = int(train_ratio * x_data.shape[0])

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]

    # shuffle training data
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    y_train = y_train[indexes]

    return (x_train, y_train), (x_test, y_test)

def metrics(y_pred, y_true):
    """
    Calucate evaluation metrics for precision, recall, and f1.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1