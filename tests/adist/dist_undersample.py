import pandas as pd
import numpy as np
import sys

usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
X = df.as_matrix()

D = 0.00005
visited = np.zeros(len(X), bool)
include = np.zeros(len(X), bool)

for i in range(len(X)):
    if not visited[i]:
        sys.stdout.write('\r%.2f%%' % (100*i/len(X)))
        sys.stdout.flush()
        dd = np.sum((X-X[i])**2, 1)
        jj = dd < D**2
        include[i] = True
        visited[i] = True
        visited[jj] = True

sys.stdout.write('\r               \r')

np.savetxt('../data/subset_train.csv', X[include], delimiter=',')
