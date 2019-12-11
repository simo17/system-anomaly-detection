from utils import metrics
from sklearn.ensemble import RandomForestClassifier

class RandomForest(object):

    def __init__(self, n_estimators, max_depth, bootstrap, oob_score, class_weight= 'balanced'):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, oob_score=oob_score, class_weight= class_weight)

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

