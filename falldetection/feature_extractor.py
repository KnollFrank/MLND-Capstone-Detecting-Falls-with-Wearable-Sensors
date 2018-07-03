import logging

import numpy as np

from falldetection.extract_features import extract_features
from falldetection.sensor_file_reader import read_sensor_file
from falldetection.window_around_maximum_total_acceleration import get_window_around_maximum_total_acceleration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:

    def extract_features(self, sensorFile):
        logger.debug('default_feature_extractor(%s)', sensorFile)
        return self.features2array(
            extract_features(
                get_window_around_maximum_total_acceleration(
                    read_sensor_file(sensorFile),
                    half_window_size=50,
                    index_error_msg=sensorFile)))

    @staticmethod
    def features2array(features):
        return np.concatenate(features.T.values)
