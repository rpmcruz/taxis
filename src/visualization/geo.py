import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

print('loading data...')
usecols = ['starting_latitude', 'starting_longitude', 'starting_timestamp',
           'revenue_class']
df = pd.read_csv('../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon', 'time', 'y']

time = [datetime.fromtimestamp(t) for t in df['time']]
df['hour'] = [t.hour for t in time]
df['workhour'] = np.logical_and(df['hour'] >= 8, df['hour'] < 20).astype(int)

limits = (
    (df['lon'].quantile(0.001), df['lon'].quantile(0.999)),
    (df['lat'].quantile(0.001), df['lat'].quantile(0.999)))


def plot_hist2d(df):
    nbins = 50
    H, xedges, yedges = np.histogram2d(df['lon'], df['lat'], nbins, limits)
    H = (H - np.min(H)) / (np.max(H) - np.min(H))
    plt.imshow(H)
    plt.xticks(
        np.linspace(0, nbins, 10),
        ['%.2f' % i for i in np.linspace(limits[0][0], limits[0][1], 10)],
        rotation=90)
    plt.yticks(
        np.linspace(0, nbins, 10),
        ['%.2f' % i for i in np.linspace(limits[1][0], limits[1][1], 10)])

for revenue in range(1, 5+1):
    plt.subplot(1, 3, 1)
    d = df[df['y'] == revenue]
    plot_hist2d(d)
    plt.title('All hours')

    for workhour in [1, 0]:
        plt.subplot(1, 3, workhour+2)
        d = df[np.logical_and(df['y'] == revenue, df['workhour'] == workhour)]
        plot_hist2d(d)
        plt.title('Work hours' if workhour else 'Leisure hours')

    plt.suptitle('Revenue class %d' % revenue)
    # plt.colorbar()
    plt.show()
