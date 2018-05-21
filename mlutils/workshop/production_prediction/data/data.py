import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data():
    here = os.path.realpath(__file__)
    here = os.path.dirname(here)
    fn = os.path.join(here, 'monthly-milk-production.csv')
    return pd.read_csv(fn, index_col='Month')


def clean_data(df):
    # convert index to a timeseries
    df.index = pd.to_datetime(df.index)
    return df


def train_test_split(df):
    train_set = df.head(156)
    test_set = df.tail(12)
    return train_set, test_set


if __name__ == "__main__":
    print(load_data())
