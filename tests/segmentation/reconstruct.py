import pandas as pd
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gini(array):
    # https://github.com/oliviaguest/gini
    array = np.sort(array)
    index = np.arange(1, array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2*index-n-1) * array)) / (n * np.sum(array)))


def dmax(array):
    array = np.sort(array)
    return np.diff(array).max()

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
D = 0.002
NDIRS = 8
NBINS = 10

i = 100  # test

# 1. neighbors of i
d = (X[:, 0] - X[i, 0])**2 + (X[:, 1] - X[i, 1])**2
jj = X[np.logical_and(d > 0, d < D**2)]

# 2. gradient of the neighbors (normalized)
djj = X[i] - jj
ndjj = djj / np.sqrt(djj[:, 0]**2 + djj[:, 1]**2)[:, np.newaxis]

# 3. map gradient to direction
# I use [-1,+1], rather than [-pi,pi] for easiness
angle = np.arctan2(ndjj[:, 1], ndjj[:, 0])/np.pi
angle = np.round(angle*NDIRS).astype(int) % NDIRS

# 4. ver qual histograma de direcção "é" uniforme
RULE = 'dmax'

min_dir = 0
min_g = np.inf
for dir in range(NDIRS):
    udjj = np.sqrt(np.sum(djj[angle == dir]**2, 1))
    if RULE == 'gini':
        g = gini(udjj)
    elif RULE == 'ks':
        H = np.histogram(udjj, NBINS, (0, D))[0]
        H = H / H.sum()
        g = kstest(H, 'uniform')[0]
    elif RULE == 'dmax':
        g = dmax(udjj)

    print(dir, g)
    if g < min_g:
        min_g = g
        min_dir = dir

if NDIRS == 4:
    sides = ['-', '/', '|', '\\']
else:
    sides = [str(i) for i in range(NDIRS)]
print('uniform side: %s' % sides[min_dir])

plt.scatter(X[:, 0], X[:, 1], 1, 'black')
colors = cm.rainbow(np.linspace(0, 1, NDIRS))
for dir in range(NDIRS):
    plt.scatter(
        jj[angle == dir][:, 0], jj[angle == dir][:, 1], 1, colors[dir],
        label=sides[dir])
plt.scatter(
    X[i, 0], X[i, 1], 100, colors[min_dir], edgecolor='black', linewidth=3)
plt.legend(title='sides')
plt.title('Same road as i')
plt.show()
