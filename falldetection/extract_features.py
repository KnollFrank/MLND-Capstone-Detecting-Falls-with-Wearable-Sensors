import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf


def extract_features(df, autocorr_num):
    features = pd.DataFrame(
        index=['min', 'max', 'mean', 'var', 'skew', 'kurtosis'] + create_autocorr_indices(autocorr_num),
        columns=df.columns,
        dtype=np.float64)

    def order_by_column(series):
        return series[features.columns].values

    def autocovariance(df, lag=1, axis=0):
        return df.apply(lambda col: acovf(col)[lag], axis=axis)

    features.loc['min', :] = order_by_column(df.min())
    features.loc['max', :] = order_by_column(df.max())
    features.loc['mean', :] = order_by_column(df.mean())
    features.loc['var', :] = order_by_column(df.var(ddof=0))
    features.loc['skew', :] = order_by_column(df.skew())
    features.loc['kurtosis', :] = order_by_column(df.kurtosis())
    # TODO: refactor
    for lag in range(1, autocorr_num + 1):
        features.loc[create_autocorr_index(lag), :] = order_by_column(autocovariance(df, lag=lag))
    return features


# TODO: refactor
def create_autocorr_indices(autocorr_num):
    return [create_autocorr_index(lag) for lag in range(1, autocorr_num + 1)]


def create_autocorr_index(lag):
    return 'autocorr_lag_' + str(lag)
