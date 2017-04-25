import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from time import time
np.random.seed(0)

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv(
    '../../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon']

df = df.ix[np.logical_and(df['lat'] > 40.63, df['lat'] < 40.64)]
df = df.ix[np.logical_and(df['lon'] > 22.935, df['lon'] < 22.945)]
df = df.sample(100000)

X = df.as_matrix()

# hyperparameters
Dnn = 0.0005
Droad = 0.0002
i0 = 101  # test

ii = np.asarray([i0])
it = 0

while True:
    print(it, len(ii))
    # 1. neighbors of ii
    tic = time()
    dd = np.ones(len(X))*np.inf
    _ii = ii[np.random.choice(len(ii), min(len(ii), 20000), False)]
    for i in _ii:
        d = (X[:, 0] - X[i, 0])**2 + (X[:, 1] - X[i, 1])**2
        dd = np.minimum(dd, d)

    jj = np.where(dd < Dnn**2)[0]
    toc = time()
    print('step 1: %.2f' % ((toc-tic)/60))

    # 2 linear regression line
    tic = time()
    _jj = jj[np.random.choice(len(jj), min(len(jj), 20000), False)]
    m = LinearRegression().fit(X[_jj, 0:1], X[_jj, 1])
    toc = time()
    print('step 2: %.2f' % ((toc-tic)/60))

    # 3. distance to linear regression
    # distance = |ax+by+c| / sqrt(a^2+b^2)
    tic = time()
    a = -m.coef_[0]
    b = 1
    c = -m.intercept_
    dd = np.abs(a*X[jj, 0] + b*X[jj, 1] + c) / np.sqrt(a**2+b**2)
    toc = time()
    print('step 3: %.2f' % ((toc-tic)/60))

    oldii = ii
    ii = jj[dd < Droad]
    if np.array_equal(oldii, ii) or i0 not in jj:
        break

    plt.scatter(X[:, 0], X[:, 1], 1, 'black')
    plt.scatter(X[ii, 0], X[ii, 1], 1, 'cyan')
    plt.scatter(
        X[i0, 0], X[i0, 1], 60, 'red', edgecolor='black', linewidth=2)
    plt.xlim(X[:, 0].min(), X[:, 0].max())
    plt.ylim(X[:, 1].min(), X[:, 1].max())
    plt.show()
    it += 1
