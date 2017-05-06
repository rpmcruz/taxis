import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import json

road_filename = '../../../data/initroads.json'
roads = []

print('loading data...')
usecols = ['starting_latitude', 'starting_longitude']
df1 = pd.read_csv(
    '../../../data/data_train_competition.csv', usecols=usecols)
df1.columns = ['lat', 'lon']

df2 = pd.read_csv(
    '../../../data/data_test_N_competition.csv', usecols=usecols)
df2.columns = ['lat', 'lon']

df = pd.concat((df1, df2))
df = df.sample(frac=0.1)
X = df.as_matrix()

X = X[X[:, 0] > 40.53]
X = X[X[:, 0] < 40.53+0.125]
X = X[X[:, 1] > 22.95]
X = X[X[:, 1] < 22.95+0.125]
print('data len: %d' % len(df))

D = 0.001


def neighbors(x):
    dd = (X[:, 0] - x[0])**2 + (X[:, 1] - x[1])**2
    jj = np.argsort(dd)
    return X[jj[1:11]]  #, dd[jj[10]]


def create_road(x):
    Xj = neighbors(x)
    model = LinearRegression().fit(Xj[:, 0:1], Xj[:, 1])
    pts = [0, 0]
    for sign in range(2):
        delta = 0
        while True:
            delta += D * (sign*2-1)
            xx = x[0] + delta
            yy = model.predict([[xx]])[0]
            dd = (X[:, 0] - xx)**2 + (X[:, 1] - yy)**2
            Xj = X[dd < D**2]
            pts[sign] = (xx, yy)
            if len(Xj) < 2:
                break
    return pts


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

print('generating roads...')
for it, i in enumerate(np.random.choice(len(X), len(X), False)):
    sys.stdout.write('\r%5.2f%% (%d)' % (100*it/len(X), len(roads)))
    sys.stdout.flush()
    dd = roads_distance(X[i])
    if len(dd) == 0 or dd.min() > D:
        # create new road
        road = create_road(X[i])
        roads.append(road)
sys.stdout.write('\r                     \r')
print('roads len: %d' % len(roads))

print('saving roads...')
with open(road_filename, 'w') as f:
    json.dump(roads, f)
