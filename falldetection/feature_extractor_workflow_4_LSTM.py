import numpy as np

from falldetection.feature_extractor import sensor_file_2_df
from falldetection.feature_extractor_4_LSTM import FeatureExtractor4LSTM
from falldetection.sensor_files_provider import SensorFilesProvider


class FeatureExtractorWorkflow4LSTM:

    def __init__(self, sensor_file_2_df, columns):
        self.sensor_file_2_df = sensor_file_2_df
        self.columns = columns

    def extract_features(self, sensorFiles):
        X, y = self.__unzip(map(self.__extract_features_4_sensorFile, sensorFiles))
        return np.array(X), np.array(y)

    def __extract_features_4_sensorFile(self, sensorFile):
        feature_extractor_4_lstm = FeatureExtractor4LSTM(self.sensor_file_2_df, self.columns)
        X, y = feature_extractor_4_lstm.extract_features(sensorFile)
        return X, [y]

    def __unzip(self, X_y_pairs):
        X, y = zip(*X_y_pairs)
        return X, y


# TODO: FeatureExtractor4LSTM hat columns und FeatureExtractorWorkflow4LSTM auch. Das ist einer zu viel, oder?
def extract_features_4_LSTM(sensor, baseDir, sensor_files_to_exclude, columns):
    sensor_files = SensorFilesProvider(baseDir, sensor, sensor_files_to_exclude).provide_sensor_files()
    features = FeatureExtractorWorkflow4LSTM(sensor_file_2_df, columns).extract_features(sensor_files)
    return features
