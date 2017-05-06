import pandas as pd
import numpy as np
import sys

N = 100  # calculate adist for N closest points
NN = 10000  # operative diameter (for memory sake)
USE_SUBSET = True

if USE_SUBSET:
    X = np.loadtxt('../data/subset_train.csv', delimiter=',')
else:
    usecols = ['starting_latitude', 'starting_longitude']
    df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
    X = df.as_matrix()

f = open('adist.csv', 'w')

for i in range(len(X)):
    sys.stdout.write('\r%.2f%%' % (100*i/len(X)))
    sys.stdout.flush()

    # due to memory constrains, define operative diameter
    jj = ((X-X[i])**2).sum(1).argsort()[:NN]
    XX = X[jj]

    dist = np.zeros((NN, NN))  # global distances
    for i in range(NN):
        dist[i] = ((XX[i] - XX)**2).sum(1)

    adist = np.zeros(N, int)
    Q = np.zeros(NN, bool)  # part-of-the-graph
    Q[i] = True
    for nn in range(N):
        j = np.amin(np.argmin(dist[Q][:, ~Q], 1))
        j = np.where(~Q)[0][j]
        Q[j] = True
        adist[nn] = jj[j]
    f.write(','.join(adist.astype(str)) + '\n')

sys.stdout.write('\r                 \r')
f.close()
