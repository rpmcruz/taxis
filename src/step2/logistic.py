import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    names = ('cluster-dist', 'cluster-class-dist')
    retr = None
    rets = None
    for colstart in names:
        print(colstart)
        usecols = [c for c in tr.columns if c.startswith(colstart)]

        Xtr = tr.as_matrix(usecols)
        ytr = tr.as_matrix(['y'])[:, 0].astype(int)
        Xts = ts.as_matrix(usecols)

        clf = GridSearchCV(
            LogisticRegression(),
            {'class_weight': ['balanced'], 'C': 10.**np.arange(-2, 7)},
            scoring, n_jobs=-1, return_train_score=False)
        clf.fit(Xtr, ytr)
        print(pd.DataFrame(clf.cv_results_))

        yptr = clf.predict(Xtr)
        ypts = clf.predict(Xts)

        retr = yptr if retr is None else np.c_[retr, yptr]
        rets = ypts if rets is None else np.c_[rets, ypts]
    return pd.DataFrame(retr, columns=names), pd.DataFrame(rets, columns=names)
