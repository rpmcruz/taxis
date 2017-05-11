import numpy as np
import sys

print('load data...')
X = np.loadtxt('../../data/subset_train.csv', delimiter=',', usecols=[1, 2])
D = np.loadtxt('../../data/adist.csv', int, delimiter=',')

print('running...')
cols = ['slope1', 'atan1', 'dist1', 'slope2', 'atan2', 'dist2']
out = np.zeros((len(X), len(cols)))

for ix, dd in enumerate(D):
    sys.stdout.write('\r%.2f%%' % (100*ix/len(X)))
    sys.stdout.flush()
    dd = dd[dd < len(X)]  # TEMP
    for k in [0, 1]:
        i = X[dd, k].argmin()
        j = X[dd, k].argmax()
        dlat = X[j, 0] - X[i, 0]
        dlon = X[j, 1] - X[i, 1]
        out[ix, 0+k*3] = dlat / dlon
        out[ix, 1+k*3] = np.arctan2(dlat, dlon)
        out[ix, 2+k*3] = np.sqrt(dlat**2 + dlon**2)

sys.stdout.write('\r                  \r')
print('saving...')
np.savetxt(
    '../../data/adist-features-dir.csv', out, delimiter=',',
    header=','.join(cols), comments='')
