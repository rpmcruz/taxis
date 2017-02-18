import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    usecols = ['idy', 'is-workhour', 'is-weekend'] + \
        [c for c in tr.columns if c.startswith('cluster-dist')]

    Xtr = tr.as_matrix(usecols)
    ytr = tr.as_matrix(['y'])[:, 0].astype(int)
    Xts = ts.as_matrix(usecols)

    clf = GridSearchCV(
        MLPClassifier(),
        {'hidden_layer_sizes': [[6, 4]],
         'alpha': 10.**np.arange(-8, -5),
         'learning_rate_init': 10.**np.arange(-3, -1)},
        scoring, n_jobs=-1, return_train_score=False)
    clf.fit(Xtr, ytr)
    print(pd.DataFrame(clf.cv_results_))

    clf = clf.best_estimator_
    clf.n_layers_ -= 1
    clf.n_outputs_ = clf.hidden_layer_sizes[-1]
    clf.hidden_layer_sizes = clf.hidden_layer_sizes[:-1]
    clf.intercepts_ = clf.intercepts_[:-1]
    clf.coefs_ = clf.coefs_[:-1]

    ztr = clf.predict(Xtr)
    zts = clf.predict(Xts)
    return pd.DataFrame({
        'nn%d' % i: ztr[:, i] for i in clf.n_outputs_}), pd.DataFrame({
        'nn%d' % i: zts[:, i] for i in clf.n_outputs_})
