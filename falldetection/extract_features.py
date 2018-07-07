import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf


def extract_features(df, autocorr_num):
    features = pd.DataFrame(columns=df.columns, dtype=np.float64)

    def add_2_features(index, df):
        def order_by_column(series):
            return series[features.columns].values

        features.loc[index, :] = order_by_column(df)

    def add_autocovariance_of_df_2_features():
        def autocovariance_of_df(lag):
            return df.apply(lambda col: acovf(col)[lag], axis='index')

        for lag in range(1, autocorr_num + 1):
            add_2_features('autocorr_lag_' + str(lag), autocovariance_of_df(lag=lag))

    add_2_features('min', df.min())
    add_2_features('max', df.max())
    add_2_features('mean', df.mean())
    add_2_features('var', df.var(ddof=0))
    add_2_features('skew', df.skew())
    add_2_features('kurtosis', df.kurtosis())
    add_autocovariance_of_df_2_features()
    return features
