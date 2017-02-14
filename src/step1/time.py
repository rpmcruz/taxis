import pandas as pd
from datetime import datetime


def run(tr, ts):
    ret = []
    for df in [tr, ts]:
        time = [datetime.fromtimestamp(t) for t in df['time']]
        ret.append(pd.DataFrame({
            'weekday': [t.weekday() for t in time],
            'month': [t.month for t in time],
            'hour': [t.hour for t in time],
        }))
    return ret
