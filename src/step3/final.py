import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    cols_except = ['y', 'time']
    usecols = [c for c in ts.columns if c not in cols_except]

    Xtr = tr.as_matrix(usecols)
    ytr = tr.as_matrix(['y'])[:, 0].astype(int)
    Xts = ts.as_matrix(usecols)

    models = [
        (GaussianNB(), {'priors': [None]}),
        (RandomForestClassifier(1000), {
            'class_weight': [None, 'balanced'],
            'min_samples_leaf': 10**np.asarray([0, 1, 2, 3, 4]),
        }),
    ]

    best_score = np.inf

    for clf, params in models:
        name = type(clf).__name__
        clf = GridSearchCV(
            clf, params, scoring, n_jobs=2, return_train_score=False)
        clf.fit(Xtr, ytr)
        print(name, pd.DataFrame(clf.cv_results_))
        if clf.best_score_ < best_score:
            best_score = clf.best_score_
            best_clf = clf

    yp = best_clf.predict(Xts)
    return pd.DataFrame(), pd.DataFrame({'y': yp})
