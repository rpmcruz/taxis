from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd
from time import time

# Experiment 1: use adist features
print('* Experiment 1')

print('load data...')
X = pd.read_csv('../../data/adist-train-features.csv').as_matrix()
y = pd.read_csv('../../data/data_train_competition.csv', usecols=[5])
y = y.as_matrix()[:, 0]

Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.20)

print('training...')
tic = time()
m = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1).fit(X, y)
toc = time()
print('%dm' % ((toc-tic)/60))
print('prediction...')
tic = time()
yp = m.predict(X)
toc = time()
print('%dm' % ((toc-tic)/60))

print('scores...')
print(cohen_kappa_score(y, yp, range(1, 6), 'quadratic'))
c = confusion_matrix(y, yp)
c = c / c.sum(1)
print(c)

# Experiment 2: use latitude,longitude features
print()
print('* Experiment 2')

print('load data...')
X = pd.read_csv('../../data/adist-train-features.csv', usecols=[2, 3])
X = X.as_matrix()
Xtr, Xts, ytr, yts = train_test_split(X, y, train_size=0.20)

print('training...')
tic = time()
m = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1).fit(X, y)
toc = time()
print('%dm' % ((toc-tic)/60))
print('prediction...')
tic = time()
yp = m.predict(X)
toc = time()
print('%dm' % ((toc-tic)/60))

print('scores...')
print(cohen_kappa_score(y, yp, range(1, 6), 'quadratic'))
c = confusion_matrix(y, yp)
c = c / c.sum(1)
print(c)
