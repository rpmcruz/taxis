import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    usecols = [c for c in tr.columns if c.startswith('cluster-dist')]

    Xtr = tr.as_matrix(usecols)
    ytr = tr.as_matrix(['y'])[:, 0].astype(int)
    Xts = ts.as_matrix(usecols)

    clf = GridSearchCV(
        LogisticRegression(),
        {'class_weight': ['balanced'], 'C': 10.**np.arange(-2, 7)},
        scoring, n_jobs=-1, return_train_score=False)
    clf.fit(Xtr, ytr)
    print('logistic', pd.DataFrame(clf.cv_results_))

    yptr = clf.predict(Xtr)
    ypts = clf.predict(Xts)
    return pd.DataFrame({'ly': yptr}), pd.DataFrame({'ly': ypts})
