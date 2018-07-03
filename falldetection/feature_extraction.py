import logging

import numpy as np
import pandas as pd

from falldetection.fall_predicate import isFall
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.window_around_maximum_total_acceleration import get_window_around_maximum_total_acceleration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def extract_all_features(sensorFiles, feature_extractor=default_feature_extractor):
    def asDataFrame(sensorFiles_features_falls):
        sensorFiles, features, falls = zip(*sensorFiles_features_falls)
        return pd.DataFrame({'sensorFile': sensorFiles, 'fall': falls, 'feature': features})

    sensorFiles_features_falls = [(sensorFile, feature_extractor(sensorFile), isFall(sensorFile)) for sensorFile in
                                  sensorFiles]
    return asDataFrame(sensorFiles_features_falls)


def extract_all_features_and_save():
    sensor_files = SensorFilesProvider(baseDir='../data/FallDataSet', sensorFile='340535.txt').provide_sensor_files()
    all_features = extract_all_features(sensor_files)
    all_features.to_csv('../data/all_features.csv')
