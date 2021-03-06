import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score
from xgboost import XGBClassifier
import numpy as np


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    cols_except = ['y', 'time', 'id']
    usecols = [c for c in ts.columns if c not in cols_except]

    for usecols in [
        #[c for c in ts.columns if c.startswith('cluster-')],
        #[c for c in ts.columns if c.startswith('cluster-dist-')],
        #[c for c in ts.columns if c.startswith('grid-')],
        #['kde'],
        #['hour', 'month', 'weekday'],
        #['ly'],
        #['lat', 'lon'],
        ['time'],
    ]:
        print()
        print()
        print('Features:')
        print(usecols)
        Xtr = tr.as_matrix(usecols)
        ytr = tr.as_matrix(['y'])[:, 0].astype(int)
        Xts = ts.as_matrix(usecols)

        models = [
            (GaussianNB(), {'priors': [None]}),
            (RandomForestClassifier(100, n_jobs=-1), {
                'class_weight': ['balanced'],
                'max_depth': [None],#np.linspace(6, 18, 5, dtype=int),
            }),
            #(XGBClassifier(), {
            #    'max_depth': np.linspace(4, 20, 5, dtype=int),
            #    'learning_rate': [0.1, 0.5],
            #}),
        ]

        best_score = np.inf

        for clf, params in models:
            name = type(clf).__name__
            clf = GridSearchCV(
                clf, params, scoring, n_jobs=1, verbose=2, refit=False, return_train_score=False)
            clf.fit(Xtr, ytr)
            print(name)
            print(pd.DataFrame(clf.cv_results_))
            if clf.best_score_ < best_score:
                best_score = clf.best_score_
                best_clf = clf

    yp = best_clf.predict(Xts)
    return pd.DataFrame(), pd.DataFrame({'y': yp})
