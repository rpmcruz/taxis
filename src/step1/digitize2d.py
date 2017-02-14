import pandas as pd
import numpy as np


def digitize2d(v1, v2, nbins, range_):
    # this will return #bins=nbins**2
    cut1 = np.linspace(range_[0][0], range_[0][1], nbins)
    cut2 = np.linspace(range_[1][0], range_[1][1], nbins)
    ret1 = np.sum(v1[:, np.newaxis] > cut1[np.newaxis, :], 1)
    ret2 = np.sum(v2[:, np.newaxis] > cut2[np.newaxis, :], 1)
    return ret1*nbins + ret2


def run(tr, ts):
    Xtr = tr.as_matrix(['lat', 'lon'])
    Xts = ts.as_matrix(['lat', 'lon'])

    range_ = (
        (Xtr[:, 0].min(), Xtr[:, 0].max()),
        (Xtr[:, 1].min(), Xtr[:, 1].max()))

    nbins = 8
    Htr = digitize2d(Xtr[:, 0], Xtr[:, 1], nbins, range_)
    Hts = digitize2d(Xts[:, 0], Xts[:, 1], nbins, range_)

    # one hot encoding
    Htr = np.asarray([[int(h == i) for h in Htr] for i in range(nbins**2)]).T
    Hts = np.asarray([[int(h == i) for h in Hts] for i in range(nbins**2)]).T

    names = ['grid-%d' % i for i in range(nbins**2)]
    return pd.DataFrame(Htr, columns=names), pd.DataFrame(Hts, columns=names)
