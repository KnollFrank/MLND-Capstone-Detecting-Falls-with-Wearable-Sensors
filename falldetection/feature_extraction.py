import logging

import numpy as np
import pandas as pd

from falldetection.extract_features import extract_features
from falldetection.fall_predicate import isFall
from falldetection.sensor_file_reader import read_sensor_file
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.window_around_maximum_total_acceleration import get_window_around_maximum_total_acceleration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def features2array(features):
    return np.concatenate(features.T.values)


def default_feature_extractor(sensorFile):
    logger.debug('default_feature_extractor(%s)', sensorFile)
    return features2array(
        extract_features(
            get_window_around_maximum_total_acceleration(
                read_sensor_file(sensorFile),
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
