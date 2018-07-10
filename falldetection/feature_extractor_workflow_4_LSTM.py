import numpy as np

from falldetection.feature_extractor_4_LSTM import FeatureExtractor4LSTM


class FeatureExtractorWorkflow4LSTM:

    def __init__(self, feature_extractor, columns):
        self.feature_extractor = feature_extractor
        self.columns = columns

    def extract_features(self, sensorFiles):
        X, y = self.__unzip(map(self.__extract_features_4_sensorFile, sensorFiles))
        return np.array(X), np.array(y)

    def __extract_features_4_sensorFile(self, sensorFile):
        feature_extractor_4_lstm = FeatureExtractor4LSTM(
            sensor_file_2_df=lambda sensor_file: self.feature_extractor(sensor_file),
            columns=self.columns)
        X, y = feature_extractor_4_lstm.extract_features(sensorFile)
        return X, [y]

    def __unzip(self, X_y_pairs):
        X, y = zip(*X_y_pairs)
        return X, y
