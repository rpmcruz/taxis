import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

usecols = ['starting_latitude', 'starting_longitude', 'revenue_class']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon', 'y']

# undersample
df = df.sample(10000)

X = df.as_matrix(['lat', 'lon'])
y = df.as_matrix(['y'])[:, 0]

print('* Dummy')

m = LogisticRegression().fit(X, y)
print(cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic'))

print('* Against random points on a map')

dists = []
for it in range(100):
    i = np.random.randint(0, len(X), 1)
    dist = list(((X-X[i])**2).sum(1))
    dists += dist

_X = np.asarray(dists)[:, np.newaxis]
_y = np.tile(y, 100)
m = LogisticRegression().fit(_X, _y)
print(cohen_kappa_score(_y, m.predict(_X), list(range(1, 6)), 'quadratic'))

print('* Against the minimum of two random points')

dists = []
for it in range(100):
    i, j = np.random.randint(0, len(X), 2)
    disti = ((X-X[i])**2).sum(1)
    distj = ((X-X[j])**2).sum(1)
    dist = [min(di, dj) for di, dj in zip(disti, distj)]
    dists += dist

_X = np.asarray(dists)[:, np.newaxis]
_y = np.tile(y, 100)
m = LogisticRegression().fit(_X, _y)
print(cohen_kappa_score(_y, m.predict(_X), list(range(1, 6)), 'quadratic'))
