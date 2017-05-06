import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
INITIAL_D = 0.002
INCR_D = 0.002
NDIRS = 18


def get_angle(x, D):
    # 1. neighbors of i
    d = (X[:, 0] - x[0])**2 + (X[:, 1] - x[1])**2
    jj = X[np.logical_and(d > 0, d < D**2)]

    # 2. gradient of the neighbors (normalized)
    djj = x - jj
    ndjj = djj / np.sqrt(djj[:, 0]**2 + djj[:, 1]**2)[:, np.newaxis]

    # 3. map gradient to direction
    # I use [-1,+1], rather than [-pi,pi] for easiness
    angles = np.arctan2(ndjj[:, 1], ndjj[:, 0])
    dirs = np.round(NDIRS*angles/np.pi).astype(int) % NDIRS

    print('dirs:', dirs)

    # 4. ver qual histograma de direcção "é" uniforme
    min_dir = 0
    min_g = np.inf
    for dir in range(NDIRS):
        print(dirs, dir)
        udjj = np.sqrt(np.sum(djj[dirs == dir]**2, 1))
        if not len(udjj):
            continue
        print('udjj:', udjj)
        g = dmax(udjj)
        if g < min_g:
            min_g = g
            min_dir = dir
    return np.pi*(min_dir+0.5)/NDIRS

i = 100  # test
angle = get_angle(X[i], INITIAL_D)

delta = INCR_D
pts = [X[i].copy(), X[i].copy()]
for sign in range(2):
    while True:
        d = delta * (2*sign-1)
        print('delta:', d)
        print('sum:', np.cos(angle)*d, np.sin(angle)*d)
        pts[sign] += np.cos(angle)*d, np.sin(angle)*d
        if get_angle(pts[sign], INITIAL_D) != angle:
            break

plt.scatter(X[:, 0], X[:, 1], 1, 'black')
plt.plot((pts[0][0], pts[1][0]), (pts[0][1], pts[1][1]), 'blue')
plt.scatter(X[i, 0], X[i, 1], 20, 'red')
plt.show()
