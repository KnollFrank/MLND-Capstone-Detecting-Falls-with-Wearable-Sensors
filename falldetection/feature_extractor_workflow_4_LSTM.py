import numpy as np

from falldetection.feature_extractor import FeatureExtractor
from falldetection.feature_extractor_4_LSTM import FeatureExtractor4LSTM
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.sensor_files_to_exclude import get_sensor_files_to_exclude_for


class FeatureExtractorWorkflow4LSTM:

    def __init__(self, featureExtractor4LSTM):
        self.featureExtractor4LSTM = featureExtractor4LSTM

    def extract_features(self, sensorFiles):
        X, y = self.__unzip(map(self.__extract_features_4_sensorFile, sensorFiles))
        return np.array(X), np.array(y)

    def __extract_features_4_sensorFile(self, sensorFile):
        X, y = self.featureExtractor4LSTM.extract_features(sensorFile)
        return X, [y]

    def __unzip(self, X_y_pairs):
        X, y = zip(*X_y_pairs)
        return X, y


def extract_features_4_LSTM(sensor, baseDir, columns):
    sensor_files = SensorFilesProvider(baseDir, sensor, get_sensor_files_to_exclude_for(sensor)).provide_sensor_files()
    features = FeatureExtractorWorkflow4LSTM(
        FeatureExtractor4LSTM(FeatureExtractor.sensor_file_2_df, columns)).extract_features(
        sensor_files)
    return features
