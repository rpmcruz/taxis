from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def run(tr, ts):
    Xtr = tr.as_matrix(['lat', 'lon'])
    Xts = ts.as_matrix(['lat', 'lon'])

    print('check outliers...')
    m = NearestNeighbors(10).fit(Xtr)

    dtr, _ = m.kneighbors(Xtr)
    dtr = np.mean(dtr[:, 1:], 1)

    dts, _ = m.kneighbors(Xts)
    dts = np.mean(dts[:, :-1], 1)

    tr_inliers = dtr < 0.02
    ts_inliers = dts < 0.02

    print('clustering all points...')
    k_all = 10
    m = KMeans(k_all)
    _Ctr = m.fit_predict(Xtr[tr_inliers])
    _Cts = m.predict(Xts[ts_inliers])

    # outliers = cluster 0
    _Ctr += 1
    Ctr = np.zeros(len(Xtr), int)
    Ctr[tr_inliers] = _Ctr

    _Cts += 1
    Cts = np.zeros(len(Xts), int)
    Cts[ts_inliers] = _Cts

    Dtr = m.transform(Xtr)
    Dts = m.transform(Xts)

    # one hot encoding
    Ctr = np.asarray([[int(c == i) for c in Ctr] for i in range(k_all+1)]).T
    Cts = np.asarray([[int(c == i) for c in Cts] for i in range(k_all+1)]).T

    Xtr_ = np.c_[Ctr, Dtr]
    Xts_ = np.c_[Cts, Dts]

    print('clustering across revenue classes...')
    k_across = 3
    y = tr.as_matrix(['y'])[:, 0]
    Dtrs = []
    Dtss = []
    for klass in range(1, 6):
        Xtr[y == klass]
        m = KMeans(k_across)
        m.fit(Xtr[np.logical_and(tr_inliers, y == klass)])
        Dtrs.append(np.amin(m.transform(Xtr), 1))
        Dtss.append(np.amin(m.transform(Xts), 1))

    Dtrs = np.asarray(Dtrs).T
    Dtss = np.asarray(Dtss).T

    Xtr_ = np.c_[Xtr_, Dtrs]
    Xts_ = np.c_[Xts_, Dtss]

    names = ['cluster-%d' % i for i in range(k_all+1)] + \
        ['cluster-dist-%d' % i for i in range(k_all)] + \
        ['cluster-class-dist-%d' % i for i in range(1, 6)]
    return pd.DataFrame(Xtr_, columns=names), pd.DataFrame(Xts_, columns=names)
