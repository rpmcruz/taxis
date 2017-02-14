from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np


# these are two heuristics often used for bandwidth selection
def scott_rule(X):
    n, d = X.shape
    return n**(-1./(d+4))


def silverman_rule(X):
    n, d = X.shape
    return (n * (d + 2) / 4.)**(-1. / (d + 4))


def run(tr, ts):
    Xtr = tr.as_matrix(['lat', 'lon'])
    Xts = ts.as_matrix(['lat', 'lon'])

    # undersampling to make these faster
    X = Xtr[np.random.choice(Xtr.shape[0], 10000, False)]

    bandwidth = scott_rule(X)
    m = KernelDensity(bandwidth).fit(X)
    Ptr = m.score_samples(Xtr)
    Pts = m.score_samples(Xts)

    names = ['kde']
    return pd.DataFrame(Ptr, columns=names), pd.DataFrame(Pts, columns=names)
