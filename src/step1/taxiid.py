import pandas as pd
import numpy as np

#
def run(tr, ts):
    Xtr = tr.as_matrix(['id', 'y'])
    Xts = ts.as_matrix(['id'])

    ids = np.unique(Xtr[:, 0])
    yy = np.zeros((len(ids), 5))
    for i, id in enumerate(ids):
        ii = Xtr[Xtr[:, 0] == id][:, 1].astype(int)
        yy[i] = np.bincount(ii, minlength=6)[1:] / len(ii)

    NA = np.bincount(Xtr[:, 1].astype(int))[1:] / len(Xtr)

    ytr = np.zeros((len(Xtr), 5))
    yts = np.zeros((len(Xts), 5))
    for i, (id, _) in enumerate(Xtr):
        if id not in ids:
            ytr[i] = NA
        else:
            ytr[i] = yy[id == ids]
    for i, id in enumerate(Xts):
        if id not in ids:
            yts[i] = NA
        else:
            yts[i] = yy[id == ids]

    return pd.DataFrame({'hid%d' % (i+1): ytr[:, i] for i in range(5)}), pd.DataFrame({'hid%d' % (i+1): yts[:, i] for i in range(5)})
