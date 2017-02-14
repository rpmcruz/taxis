from time import time
import importlib
import pandas as pd
from datetime import datetime
import os

print('# step 0')
usecols = [
    'starting_latitude', 'starting_longitude', 'starting_timestamp',
    'revenue_class']
tr = pd.read_csv('../data/data_train_competition.csv', usecols=usecols)
tr.columns = ['lat', 'lon', 'time', 'y']

usecols = usecols[:-1]
ts = pd.read_csv('../data/data_test_N_competition.csv', usecols=usecols)
ts.columns = ['lat', 'lon', 'time']


steps = []
for step in range(1, 99):
    stepdir = 'step%d' % step
    if not os.path.exists(stepdir):
        break
    substeps = [os.path.join(stepdir, s) for s in os.listdir(stepdir)
                if s.endswith('.py') and not s.startswith('__')]
    steps.append(substeps)

dependency_obsolete = False

for stepi, substeps in enumerate(steps):
    outdated = False
    substep_tr = []
    substep_ts = []

    for substepi, substep in enumerate(substeps):
        tic = time()
        substep_filename = substep.replace('/', '')[:-3]
        tr_filename = '../data/features/%s-tr.csv' % substep_filename
        ts_filename = '../data/features/%s-ts.csv' % substep_filename
        substep_name = substep.split('/')[-1][:-3]
        module_name = substep.replace('/', '.')[:-3]
        recreate = dependency_obsolete or not os.path.exists(tr_filename) or \
            os.path.getmtime(substep) > os.path.getmtime(tr_filename)
        recreate_str = '[recreate]' if recreate else '[exists]'
        progress = stepi+1, substepi+1, substep_name, recreate_str
        print('\n\n# step %d.%d %-20s %10s' % progress)
        if recreate:
            obsolete = True
            _tr, _ts = importlib.import_module(module_name).run(tr, ts)
            _tr.to_csv(tr_filename, index=False)
            _ts.to_csv(ts_filename, index=False)
        else:
            _tr, _ts = pd.read_csv(tr_filename), pd.read_csv(ts_filename)
        substep_tr.append(_tr)
        substep_ts.append(_ts)
        toc = time()
        print('[elapsed time: %dm]' % int((toc-tic)/60))
    if outdated:
        dependency_obsolete = True
    tr = pd.concat(substep_tr + [tr], 1)
    ts = pd.concat(substep_ts + [ts], 1)

t = datetime.now().strftime('%Y-%m-%d-%H-%M')
yp = pd.read_csv('../data/data_ts_competition.csv', usecols=['ID'])
yp['revenue_class'] = ts['y']
filename = '../data/yp/submission-%s.csv' % t
print('writing results to %s...' % filename)
yp.to_csv(filename, index=False)
