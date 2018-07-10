import numpy as np

from falldetection.feature_extractor import FeatureExtractor
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.sensor_files_to_exclude import get_sensor_files_to_exclude_for
from falldetection.time_series_extractor import TimeSeriesExtractor


class TimeSeriesExtractorWorkflow:

    def __init__(self, timeSeriesExtractor):
        self.timeSeriesExtractor = timeSeriesExtractor

    def extract_time_series(self, sensorFiles):
        X, y = self.__unzip(map(self.__extract_time_series_4_sensorFile, sensorFiles))
        return np.array(X), np.array(y)

    def __extract_time_series_4_sensorFile(self, sensorFile):
        X, y = self.timeSeriesExtractor.extract_time_series(sensorFile)
        return X, [y]

    def __unzip(self, X_y_pairs):
        X, y = zip(*X_y_pairs)
        return X, y


def extract_time_series(sensor, baseDir, columns):
    sensor_files = SensorFilesProvider(baseDir, sensor, get_sensor_files_to_exclude_for(sensor)).provide_sensor_files()
    time_series = TimeSeriesExtractorWorkflow(
        sensor_file_2_df=TimeSeriesExtractor(FeatureExtractor.sensor_file_2_df, columns)).extract_time_series(
        columns=sensor_files)
    return time_series
