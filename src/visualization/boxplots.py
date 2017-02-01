import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

print('loading data...')
usecols = ['starting_latitude', 'starting_longitude', 'starting_timestamp',
           'revenue_class']
df = pd.read_csv('../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon', 'time', 'y']

def discretize_two_features(v1, v2, nbins):
    # this will return #bins=nbins**2
    cut1 = np.linspace(v1.min(), v1.max(), nbins)
    cut2 = np.linspace(v2.min(), v2.max(), nbins)
    ret1 = np.sum(v1[:, np.newaxis] > cut1[np.newaxis, :], 1)
    ret2 = np.sum(v2[:, np.newaxis] > cut2[np.newaxis, :], 1)
    return ret1*nbins + ret2

print('creating features...')
time = [datetime.fromtimestamp(t) for t in df['time']]
df['weekday'] = [t.weekday() for t in time]
df['month'] = [t.month for t in time]
df['hour']  = [t.hour for t in time]
df['pos'] = discretize_two_features(df['lat'], df['lon'], 8)

print('displaying...')
for feature in ['weekday', 'month', 'pos']:
    axes = df.groupby(feature).boxplot(
        column='y', grid=False, return_type='axes')
    for _, ax in axes.items():
        ax.get_xaxis().set_visible(False)
        ax.set_yticks(np.arange(1, 6))
    plt.suptitle(feature)
    plt.show()

