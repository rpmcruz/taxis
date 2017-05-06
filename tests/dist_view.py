import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('load data...')
usecols = ['starting_latitude', 'starting_longitude']
df = pd.read_csv(
    '../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon']

df = df.ix[np.logical_and(df['lat'] > 40.63, df['lat'] < 40.635)]
df = df.ix[np.logical_and(df['lon'] > 22.935, df['lon'] < 22.940)]

X = df.as_matrix()

D = 0.00005
i = 200  # test

# 1. neighbors of i
d = (X[:, 0] - X[i, 0])**2 + (X[:, 1] - X[i, 1])**2
jj = X[np.logical_and(d > 0, d < D**2)]

plt.scatter(X[:, 0], X[:, 1], 1, 'black')
plt.scatter(jj[:, 0], jj[:, 1], 1, 'blue')
plt.scatter(X[i, 0], X[i, 1], 1, 'red')
plt.legend(title='sides')
plt.title('Same road as i')
plt.show()
