import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# http://tinyurl.com/mbhn3t6

road_filename = '../../../data/roads.json'

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv(
    '../../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon']
df = df.sample(frac=0.01)

Xtr = df.as_matrix(['lat', 'lon'])

with open(road_filename, 'r') as f:
    roads = json.load(f)

print('calculate distances...')
roads_dist = np.zeros((len(roads), len(Xtr)))

for roadi, road in enumerate(roads):
    sys.stdout.write('\r%4.1f' % (100*roadi/len(roads)))
    sys.stdout.flush()
    assert len(road) > 1
    road = np.asarray(road)

    A = road[:-1][:, :, np.newaxis]
    B = road[1:][:, :, np.newaxis]

    C = Xtr
    C = np.rollaxis(C, 1)[np.newaxis, :, :]
    C = np.repeat(C, len(A), 0)

    AB = B-A
    AC = C-A
    cross = AB[:, 0, :]*AC[:, 1, :] - AB[:, 1, :]*AC[:, 0, :]
    distAB = np.sqrt((A[:, 0, :]-B[:, 0, :])**2 + (A[:, 1, :]-B[:, 1, :])**2)
    distBC = np.sqrt((B[:, 0, :]-C[:, 0, :])**2 + (B[:, 1, :]-C[:, 1, :])**2)
    distAC = np.sqrt((A[:, 0, :]-C[:, 0, :])**2 + (A[:, 1, :]-C[:, 1, :])**2)

    dist = np.abs(cross / distAB)

    # check if outside the segment
    BC = C-B
    dot1 = AB[:, 0, :]*BC[:, 0, :] + AB[:, 1, :]*BC[:, 1, :]
    dot2 = (-AB[:, 0, :])*AC[:, 0, :] + (-AB[:, 1, :])*AC[:, 1, :]

    res = \
        (dot1 >= 0)*distBC + \
        (dot2 >= 0)*distAC + \
        np.logical_and(dot1 < 0, dot2 < 0)*dist
    roads_dist[roadi] = np.min(res, 0)

sys.stdout.write('\r            \r')

no_road = np.array(np.repeat(0.002, len(Xtr)), ndmin=2)
roads_dist = np.r_[roads_dist, no_road]
roadss = np.argmin(roads_dist, 0)

print('plot...')
colors = cm.Accent(np.linspace(0, 1, len(roads)))
black = np.array((0, 0, 0, 1), ndmin=2)
colors = np.r_[colors, black]
mycolors = [colors[i] for i in roadss]
plt.scatter(Xtr[:, 0], Xtr[:, 1], 3, mycolors)
plt.show()
