from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
from time import time

print('load target variable...')
y = pd.read_csv('../../data/data_train_competition.csv', usecols=[5])
y = y.as_matrix()[:, 0]

def train_and_score(X):
    Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.20)

    print('training...')
    tic = time()
    m = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1)
    m.fit(Xtr, ytr)
    toc = time()
    print('%dm' % ((toc-tic)/60))
    print('prediction...')
    tic = time()
    # random forest predict requires too much memory: splitting in parts
    yp = np.zeros(len(Xts))
    for _, ix in KFold(10).split(Xts):
        yp[ix] = m.predict(Xts[ix])
    toc = time()
    print('%dm' % ((toc-tic)/60))

    print('scores...')
    print(cohen_kappa_score(yts, yp, range(1, 6), 'quadratic'))
    c = confusion_matrix(yts, yp)
    c = c / c.sum(1)
    print(c)

# Experiment 1: use adist features
print('* Experiment 1')

print('load data...')
X1 = pd.read_csv('../../data/adist-train-features.csv').as_matrix()
train_and_score(X1)

# Experiment 2: use latitude,longitude features
print()
print('* Experiment 2')

print('load data...')
X = pd.read_csv('../../data/adist-train-features.csv', usecols=[2, 3])
X = X.as_matrix()

for x in np.percentile(X[:, 0], range(10, 100, 10)):
    X = np.c_[X, X[:, 0] - x]
for x in np.percentile(X[:, 1], range(10, 100, 10)):
    X = np.c_[X, X[:, 1] - x]
X2 = X
train_and_score(X)

# Experiment 3: combining both feature sets
print()
print('* Experiment 3')

X = np.c_[X1, X2]
train_and_score(X)
