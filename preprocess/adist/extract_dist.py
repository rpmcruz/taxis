import numpy as np
import sys

print('load data...')
X = np.loadtxt('../../data/subset_train.csv', delimiter=',', usecols=[1, 2])
D = np.loadtxt('../../data/adist.csv', int, delimiter=',')

print('running...')
percentiles = np.arange(0, 110, 10)
out = np.zeros((len(X), len(percentiles)))

for ix, dd in enumerate(D):
    sys.stdout.write('\r%.2f%%' % (100*ix/len(X)))
    sys.stdout.flush()

    dd1 = np.tile(X[dd], (len(dd), 1))
    dd2 = np.repeat(X[dd], len(dd), 0)
    ddd = np.sqrt(np.sum((dd1-dd2)**2, 1))

    out[ix] = np.percentile(ddd, percentiles)

sys.stdout.write('\r                  \r')
print('saving...')
np.savetxt(
    '../../data/adist-features-dist.csv', out, delimiter=',',
    header=','.join(['distp%d' % p for p in percentiles]), comments='')
