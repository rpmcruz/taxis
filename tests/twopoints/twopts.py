import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import sys


def scoring(m, X, y):
    return cohen_kappa_score(y, m.predict(X), list(range(1, 6)), 'quadratic')

usecols = ['starting_latitude', 'starting_longitude', 'revenue_class']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon', 'y']

# undersample
df = df.sample(10000)

X = df.as_matrix(['lat', 'lon'])
y = df.as_matrix(['y'])[:, 0]

print('* Dummy')

print('training with %d variables' % X.shape[1])
m = LogisticRegression().fit(X, y)
m = GridSearchCV(
    LogisticRegression(),
    {'class_weight': ['balanced'], 'C': 10.**np.arange(-2, 7)},
    scoring, n_jobs=-1, return_train_score=False)
m.fit(X, y)
print(pd.DataFrame(m.cv_results_))

print('* Against random reference points on a map')

# 1. random reference points
xx, yy = np.mgrid[
    X[:, 0].min():X[:, 0].max():75j,
    X[:, 1].min():X[:, 1].max():75j
]

xx = xx.flatten()
yy = yy.flatten()

dists = []
for pt in zip(xx, yy):
    dist = list(((X-pt)**2).sum(1))
    dists.append(dist)
_X = np.asarray(dists).T

# 2. reference points selection
m = LogisticRegression('l1', C=1e0).fit(_X, y)
i = np.nonzero(m.coef_.sum(0))[0]
print(len(i)/len(y))
_X = _X[:, i]

# 3. final model
print('training with %d variables' % _X.shape[1])
m = GridSearchCV(
    LogisticRegression(),
    {'class_weight': ['balanced'], 'C': 10.**np.arange(-2, 7)},
    scoring, n_jobs=-1, return_train_score=False)
m.fit(_X, y)
print(pd.DataFrame(m.cv_results_))

print('* Against the minimum of two random points')

# 1. random two comparison points
xx, yy = np.mgrid[
    X[:, 0].min():X[:, 0].max():17j,
    X[:, 1].min():X[:, 1].max():17j
]

xx = xx.flatten()
yy = yy.flatten()

dists = []
for i1, pt1 in enumerate(zip(xx, yy)):
    sys.stdout.write('\r%3.1f%%' % (100*i1/len(xx)))
    sys.stdout.flush()
    for i2, pt2 in enumerate(zip(xx, yy)):
        if i2 >= i1:
            disti = ((X-pt1)**2).sum(1)
            distj = ((X-pt2)**2).sum(1)
            dist = [min(di, dj) for di, dj in zip(disti, distj)]
            dists.append(dist)
_X = np.asarray(dists).T
sys.stdout.write('\r      \r')

# 2. reference points selection
m = LogisticRegression('l1', C=1e-1).fit(_X, y)
i = np.nonzero(m.coef_.sum(0))[0]
print(len(i)/len(y))
_X = _X[:, i]

# 3. final model
print('training with %d variables' % _X.shape[1])
m = GridSearchCV(
    LogisticRegression(),
    {'class_weight': ['balanced'], 'C': 10.**np.arange(-2, 7)},
    scoring, n_jobs=-1, return_train_score=False)
m.fit(_X, y)
print(pd.DataFrame(m.cv_results_))
