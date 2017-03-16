from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import matplotlib.cbook
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

print('loading data...')
usecols = ['starting_latitude', 'starting_longitude', 'starting_timestamp',
           'revenue_class']
df = pd.read_csv('../../data/data_train_competition.csv', usecols=usecols)
df.columns = ['lat', 'lon', 'time', 'y']

# remove outliers
'''
df = df[df['lon'] > df['lon'].quantile(0.001)]
df = df[df['lat'] > df['lat'].quantile(0.001)]
df = df[df['lon'] < df['lon'].quantile(0.999)]
df = df[df['lat'] < df['lat'].quantile(0.999)]
'''

colors = [(1, 0, 0, 0), (1, 0, 0, 1)]
alpha_cm = LinearSegmentedColormap.from_list('alpha_cm', colors)

limits = (
    (df['lon'].quantile(0.001), df['lon'].quantile(0.999)),
    (df['lat'].quantile(0.001), df['lat'].quantile(0.999)))

m = Basemap(
    limits[0][0], limits[1][0], limits[0][1], limits[1][1],
    projection='merc', resolution='h', area_thresh=0.01)

# histogram 2D
for revenue in range(1, 5+1):
    d = df[df['y'] == revenue]
    m.drawcoastlines()

    H, xedges, yedges = np.histogram2d(d['lon'], d['lat'], 20, limits)
    H = (H - np.min(H)) / (np.max(H) - np.min(H))
    X, Y = np.meshgrid(xedges, yedges)
    m.imshow(H, cmap=alpha_cm)
    plt.colorbar()
    plt.title('Revenue class %d' % revenue)
    plt.show()

'''
print('density estimation...')
kde = gaussian_kde(df.as_matrix(['lon', 'lat']).T)
xx, yy = np.mgrid[
    df['lon'].min():df['lon'].max():25j,
    df['lat'].min():df['lat'].max():25j]
pos = np.vstack([xx.ravel(), yy.ravel()])
f = kde(pos).T.reshape(xx.shape)
'''
