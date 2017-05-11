import numpy as np
import pandas as pd
import sys

print('load data...')
X = np.loadtxt('../../data/subset_train.csv', delimiter=',', usecols=[0])
D = np.loadtxt('../../data/adist.csv', int, delimiter=',')

usecols = ['revenue_class']
df = pd.read_csv('../../data/data_train_competition.csv', usecols=usecols)
Y = df.as_matrix()[:, 0].astype(int)

print('running...')
out = np.zeros((len(X), 5))

for ix, dd in enumerate(D):
    sys.stdout.write('\r%.2f%%' % (100*ix/len(X)))
    sys.stdout.flush()
    dd = dd[dd < len(X)]  # TEMP
    out[ix] = np.bincount(Y[dd]-1, minlength=5) / len(dd)

sys.stdout.write('\r                  \r')
print('saving...')
np.savetxt(
    '../../data/adist-features-yhist.csv', out, delimiter=',',
    header='hy1,hy2,hy3,hy4,hy5', comments='')
