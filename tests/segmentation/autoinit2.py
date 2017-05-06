import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import json

road_filename = 'testroads3.json'
roads = []

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv(
    '../../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon']

# df = df.ix[np.logical_and(df['lat'] > 40.63, df['lat'] < 40.64)]
# df = df.ix[np.logical_and(df['lon'] > 22.935, df['lon'] < 22.945)]
df = df.sample(100000)

X = df.as_matrix()

# hyperparameters
Dnn = 0.0005
Dclose = 0.0003
Droad = 0.0002


def roads_distance(x):
    roads_dist = np.zeros(len(roads))
    for roadi, road in enumerate(roads):
        assert len(road) > 1
        road = np.asarray(road)

        A = np.rollaxis(road[:-1], 1)
        B = np.rollaxis(road[1:], 1)
        C = x[:, np.newaxis]

        AB = B-A
        AC = C-A
        cross = AB[0]*AC[1] - AB[1]*AC[0]
        distAB = np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
        distBC = np.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
        distAC = np.sqrt((A[0]-C[0])**2 + (A[1]-C[1])**2)

        distAB += 0.0001  # avoid zero division
        dist = np.abs(cross / distAB)

        # check if outside the segment
        BC = C-B
        dot1 = AB[0]*BC[0] + AB[1]*BC[1]
        dot2 = (-AB[0])*AC[0] + (-AB[1])*AC[1]

        res = \
            (dot1 >= 0)*distBC + \
            (dot2 >= 0)*distAC + \
            np.logical_and(dot1 < 0, dot2 < 0)*dist
        roads_dist[roadi] = np.min(res)
    return roads_dist


def create_road(i0):
    ii = [i0]
    newii = ii
    it = 0
    while True:
        # 1. neighbors of ii
        dd = np.ones(len(X))*np.inf
        for i in newii:
            d = (X[:, 0] - X[i, 0])**2 + (X[:, 1] - X[i, 1])**2
            dd = np.minimum(dd, d)
        dd[ii] = 0

        jj = np.where(dd < Dnn**2)[0]

        # 2 linear regression line
        _jj = jj[np.random.choice(len(jj), min(len(jj), 20000), False)]
        m = LinearRegression(fit_intercept=False)
        m.fit(X[_jj, 0:1] - X[i0, 0], X[_jj, 1] - X[i0, 1])

        # 3. distance to linear regression
        # distance = |ax+by+c| / sqrt(a^2+b^2)
        a = -m.coef_[0]
        b = 1
        c = 0
        dd = np.abs(a*(X[jj, 0]-X[i0, 0]) + b*(X[jj, 1]-X[i0, 1]) + c) / \
            np.sqrt(a**2+b**2)

        oldii = ii
        ii = np.union1d(ii, jj[dd < Droad])
        newii = np.array([i for i in ii if i not in oldii])
        if len(newii) == 0:
            if it <= 1:
                return None
            x0 = X[ii, 0].min()
            x1 = X[ii, 0].max()
            return (x0, m.predict([[x0]])[0]), ((x1, m.predict([[x1]])[0]))
        it += 1

print('generating roads...')
for it, i in enumerate(np.random.choice(len(X), len(X), False)):
    sys.stdout.write('\r%5.2f%% (%d)' % (100*it/len(X), len(roads)))
    sys.stdout.flush()
    dd = roads_distance(X[i])
    if len(dd) == 0 or dd.min() > Dclose:
        # create new road
        road = create_road(i)
        if road:
            roads.append(road)
sys.stdout.write('\r                     \r')
print('roads len: %d' % len(roads))

print('saving roads...')
with open(road_filename, 'w') as f:
    json.dump(roads, f)
