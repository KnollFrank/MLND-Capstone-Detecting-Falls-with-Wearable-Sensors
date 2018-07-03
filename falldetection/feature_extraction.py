import logging
import os
import re

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def slice_with_window(df, window_center_index, half_window_size, index_error_msg=""):
    lower_bound_inclusive = window_center_index - half_window_size
    upper_bound_exclusive = window_center_index + half_window_size + 1
    if not (0 <= lower_bound_inclusive < len(df) and 0 <= upper_bound_exclusive <= len(df)):
        raise IndexError(
            "{}: not (0 <= {} < {} and 0 <= {} <= {})".format(index_error_msg, lower_bound_inclusive, len(df),
                                                              upper_bound_exclusive, len(df)))
    return df.iloc[lower_bound_inclusive:upper_bound_exclusive]


def get_index_of_maximum_total_acceleration(df):
    squared_total_acceleration = df['Acc_X'] ** 2 + df['Acc_Y'] ** 2 + df['Acc_Z'] ** 2
    return squared_total_acceleration.idxmax()


def get_window_around_maximum_total_acceleration(df, half_window_size, index_error_msg=""):
    return slice_with_window(
        df,
        window_center_index=get_index_of_maximum_total_acceleration(df),
        half_window_size=half_window_size,
        index_error_msg=index_error_msg)


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


def features2array(features):
    return np.concatenate(features.T.values)


def default_feature_extractor(sensorFile):
    logger.debug('default_feature_extractor(%s)', sensorFile)
    df = pd.read_csv(
        sensorFile,
        skiprows=4,
        sep='\t',
        usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
    return features2array(
        extract_features(
            get_window_around_maximum_total_acceleration(
                df,
                half_window_size=50,
                index_error_msg=sensorFile)))


def default_sensor_files_provider(baseDir, sensorFile):
    for root, dirs, files in os.walk(baseDir):
        for file in files:
            if file == sensorFile:
                yield os.path.join(root, file)


def extract_all_features(sensorFiles, feature_extractor=default_feature_extractor):
    def asDataFrame(sensorFiles_and_features):
        sensorFiles, features = zip(*sensorFiles_and_features)
        return pd.DataFrame({'sensorFile': sensorFiles, 'feature': features})

    sensorFiles_and_features = [(sensorFile, feature_extractor(sensorFile)) for sensorFile in sensorFiles]
    return asDataFrame(sensorFiles_and_features)


def isFall(sensorFile):
    eight_or_nine = re.search('Testler Export/(8|9)', sensorFile).group(1)
    return eight_or_nine == '9'
