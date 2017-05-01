import pandas as pd
import numpy as np
import sys

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
X = df.as_matrix()

print('calculate distances...')
nn = 1000
mdists = np.zeros((len(X), nn), int)
for i, x in enumerate(X):
    sys.stdout.write('\r%.3f%%' % (100*i/len(X)))
    d = np.sum((X-x)**2, 1)
    mdists[i] = np.argsort(d)[1:nn+1]

sys.stdout.write('\r                \r')

np.savetxt('mdists.csv', mdists, '%d', ',')
