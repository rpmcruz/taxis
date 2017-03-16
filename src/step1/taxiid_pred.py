import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import cohen_kappa_score
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    usecols = ['id']

    Xtr = tr.as_matrix(usecols)
    ytr = tr.as_matrix(['y'])[:, 0].astype(int)
    Xts = ts.as_matrix(usecols)

    enc = OneHotEncoder(handle_unknown='ignore')
    Xtr = enc.fit_transform(Xtr)
    Xts = enc.transform(Xts)

    clf = GridSearchCV(
        RandomForestClassifier(100), {
            'class_weight': ['balanced'],
            'max_depth': (10.**np.arange(1, 4).astype(int)),
        },
        scoring, n_jobs=-1, return_train_score=False)
    clf.fit(Xtr, ytr)
    print(pd.DataFrame(clf.cv_results_))

    yptr = clf.predict(Xtr)
    ypts = clf.predict(Xts)
    return pd.DataFrame({'idy': yptr}), pd.DataFrame({'idy': ypts})
