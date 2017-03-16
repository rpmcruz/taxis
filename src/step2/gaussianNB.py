import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
import numpy as np



def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')


def run(tr, ts):
    
    #tr = tr[tr['grid-30']==1]
    #ts = ts[ts['grid-30']==1]
            
    usecols = [c for c in tr.columns if c.startswith('cluster30-dist')] + \
              ['hid1', 'hid2', 'hid3', 'hid4', 'hid5', 'sin.hour', 'cos.hour']
        
    Xtr = tr.as_matrix(usecols)
    ytr = tr.as_matrix(['y'])[:, 0].astype(int)
    Xts = ts.as_matrix(usecols)

    clf = GaussianNB()
    clf.fit(Xtr, ytr)
    print('GaussianNB', scoring(clf, Xtr, ytr))

    yptr = clf.predict(Xtr)
    ypts = clf.predict(Xts)
    return pd.DataFrame({'NBy': yptr}), pd.DataFrame({'NBy': ypts})
