import numpy as np
import pandas as pd


def extract_features(df):
    features = pd.DataFrame(
        index=['min', 'max', 'mean', 'var', 'skew', 'kurtosis'],
        columns=df.columns,
        dtype=np.float64)

    def order_by_column(series):
        return series[features.columns].values

    features.loc['min', :] = order_by_column(df.min())
    features.loc['max', :] = order_by_column(df.max())
    features.loc['mean', :] = order_by_column(df.mean())
    features.loc['var', :] = order_by_column(df.var(ddof=0))
    features.loc['skew', :] = order_by_column(df.skew())
    features.loc['kurtosis', :] = order_by_column(df.kurtosis())
    return features
