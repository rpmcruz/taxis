import pandas as pd
import numpy as np
import os
import sys

print('load data...')
Xsubset = np.loadtxt(
    '../../data/subset_train.csv', delimiter=',', usecols=[1, 2])
Xtr = np.loadtxt(
    '../../data/data_train_competition.csv', delimiter=',', usecols=[2, 3],
    skiprows=1)
Xts = np.loadtxt(
    '../../data/data_test_N_competition.csv', delimiter=',', usecols=[2, 3],
    skiprows=1)

df = []
files = os.listdir('../../data')
for file in files:
    if file.startswith('adist-features-'):
        df.append(pd.read_csv('../../data/' + file))
df = pd.concat(df, 1)

for name, X in (('train', Xtr), ('test', Xts)):
    print(name)
    out = np.zeros((len(X), df.shape[1]))
    for i in range(len(X)):
        sys.stdout.write('\r%.2f%%' % (100*i/len(X)))
        sys.stdout.flush()
        ix = np.argmin(np.sum((X[i]-Xsubset)**2, 1))
        out[i] = df.ix[ix]
    sys.stdout.write('\r                  \r')
    np.savetxt(
        '../../data/adist-%s-features.csv' % name, out, delimiter=',',
        header=','.join(df.columns), comments='')
