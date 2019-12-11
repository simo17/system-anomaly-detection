from utils import metrics
from sklearn.linear_model import LogisticRegression

class LogisticRegression(object):

    def __init__(self, random_state, solver, multi_class, max_iter):
        self.model = LogisticRegression(random_state=random_state, solver=solver, multi_class=multi_class, max_iter=max_iter)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def evaluate(self, x, y_true):
        y_pred = self.model.predict(x)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
