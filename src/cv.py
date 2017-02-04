from datetime import datetime
import pandas as pd
from score import quadratic_weighted_kappa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

KNN = False
FOREST = False
LINEAR = True
LOGISTIC = False

print('loading data...')
usecols = ['taxi_id', 'starting_latitude', 'starting_longitude',
           'starting_timestamp', 'revenue_class']
df = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
df.columns = ['taxi', 'lat', 'lon', 'time', 'y']

print('creating features...')
time = [datetime.fromtimestamp(t) for t in df['time']]
df['weekday'] = [t.weekday() for t in time]
df['month'] = [t.month for t in time]
df['hour'] = [t.hour for t in time]
'''
i = 0
for lat_quantile in [0.30, 0.50, 0.70]:
    for lon_quantile in [0.30, 0.50, 0.70]:
        df['lat.%d' % i] = df['lat'] - df['lat'].quantile(lat_quantile)
        df['lon.%d' % i] = df['lon'] - df['lon'].quantile(lon_quantile)
        i += 1
'''


def scoring(m, X, y):
    yp = m.predict(X)
    return quadratic_weighted_kappa(y, yp, 1, 5)

X = df.as_matrix([c for c in df.columns if c != 'y'])
y = df.as_matrix(['y'])[:, 0]

rescols = ['mean_fit_time', 'params', 'mean_test_score', 'std_test_score',
           'rank_test_score']

print('dummy classifier...')
res = GridSearchCV(
    DummyClassifier(), {'strategy': ['stratified', 'most_frequent']}, scoring,
    n_jobs=-1, refit=False).fit(X, y).cv_results_
res = pd.DataFrame(res)[rescols]
res.to_csv('cv-dummy.csv', index=False)
print(res)

if KNN:
    print('nearest neighbors...')
    res = GridSearchCV(
        KNeighborsClassifier(), {'n_neighbors': [5, 50]}, scoring,
        n_jobs=-1, refit=False, verbose=2).fit(X, y).cv_results_
    res = pd.DataFrame(res)[rescols]
    res.to_csv('cv-knn.csv', index=False)
    print(res)

if FOREST:
    print('random forest...')
    res = GridSearchCV(
        RandomForestClassifier(100),
        {'class_weight': [None, 'balanced'], 'max_depth': [3, 5, 7]}, scoring,
        n_jobs=1, refit=False, verbose=2).fit(X, y).cv_results_
    res = pd.DataFrame(res)[rescols]
    res.to_csv('cv-forest.csv', index=False)
    print(res)

if LINEAR:
    print('linear regression...')
    from sklearn.linear_model import Ridge

    class MyLinear(Ridge):
        def predict(self, X):
            yp = np.round(super().predict(X)).astype(int)
            yp[yp < 1] = 1
            yp[yp > 5] = 5
            return yp

    class_weight = len(y) / (5*np.bincount(y)[1:])
    w = [class_weight[l-1] for l in y]

    res = GridSearchCV(
        MyLinear(),
        {'alpha': [1e-2, 1e-3, 1e-4, 1e-5]}, scoring,
        {'sample_weight': w}, 1, refit=False, verbose=2).fit(X, y).cv_results_
    res = pd.DataFrame(res)[rescols]
    res.to_csv('cv-logistic.csv', index=False)
    print(res)

if LOGISTIC:
    print('logistic regression...')
    res = GridSearchCV(
        LogisticRegression(),
        {'class_weight': [None, 'balanced'], 'C': [0.01, 1, 100]}, scoring,
        n_jobs=-1, refit=False, verbose=2).fit(X, y).cv_results_
    res = pd.DataFrame(res)[rescols]
    res.to_csv('cv-logistic.csv', index=False)
    print(res)
