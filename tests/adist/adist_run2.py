import pandas as pd
import numpy as np
import sys

usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
X = df.as_matrix()

N = 100  # calculate adist for N closest points
NN = 1000


dists = {}
access = []


def get_dist(i):
    access.append(i)
    if len(access) >= NN:
        j = access.pop(0)
        if j in dists:
            del dists[j]
    if i not in dists:
        dists[i] = ((X-X[i])**2).sum(1)
    return dists[i]


f = open('adist2.csv', 'w')

for i in range(len(X)):
    sys.stdout.write('\r%.4f%%' % (100*i/len(X)))
    sys.stdout.flush()

    adist = np.zeros(N, int)
    Qb = np.zeros(len(X), bool)  # part-of-the-graph
    Qb[i] = True
    Ql = [i]
    for nn in range(N):
        j = np.amin([np.argmin(get_dist(q)[~Qb]) for q in Ql])
        j = np.where(~Qb)[0][j]
        Qb[j] = True
        Ql.append(j)
        adist[nn] = j
    f.write(','.join(adist.astype(str)) + '\n')

sys.stdout.write('\r                 \r')
f.close()
