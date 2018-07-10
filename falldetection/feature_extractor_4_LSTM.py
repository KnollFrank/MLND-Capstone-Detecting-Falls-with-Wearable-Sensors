from falldetection.extract_features_4_LSTM import extract_features_4_LSTM
from falldetection.fall_predicate import isFall


class FeatureExtractor4LSTM:

    def __init__(self, sensor_file_2_df, columns) -> None:
        self.sensor_file_2_df = sensor_file_2_df
        self.columns = columns

    def extract_features(self, sensorFile):
        X = extract_features_4_LSTM(self.sensor_file_2_df(sensorFile), self.columns)
        y = isFall(sensorFile)
        return X, y
