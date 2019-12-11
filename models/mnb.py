from sklearn.naive_bayes import MultinomialNB
from utils import metrics

class MNB(object):

    def __init__(self):
        self.model = MultinomialNB()

    def fit(self, x, y):
        self.model.fit(X, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def evaluate(self, x, y_true):
        y_pred = self.model.predict(x)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1