import pandas as pd
from datetime import datetime


# holidays
def parse_date(d):
    return datetime.strptime(d, '%b %d').replace(year=2017).date()
holidays = pd.read_table(
    '../data/holidays-gr.tsv', comment='#', parse_dates=[0],
    date_parser=parse_date)
holidays = holidays[holidays['type'] != 'Season']['date']


def gen_time(df):
    time = [datetime.fromtimestamp(t) for t in df['time']]
    return pd.DataFrame({
        #'weekday': [t.weekday() for t in time],
        #'month': [t.month for t in time],
        #'hour': [t.hour for t in time],
        'is-weekend': [
            int(t.weekday() >= 5 or t.date() in holidays) for t in time],
        'is-workhour': [int(8 <= t.hour < 20) for t in time],
    })


def run(tr, ts):
    return gen_time(tr), gen_time(ts)
