from falldetection.extract_time_series import extract_time_series
from falldetection.fall_predicate import isFall


class TimeSeriesExtractor:

    def __init__(self, sensor_file_2_df, columns) -> None:
        self.sensor_file_2_df = sensor_file_2_df
        self.columns = columns

    def extract_time_series(self, sensorFile):
        X = extract_time_series(self.sensor_file_2_df(sensorFile), self.columns)
        y = isFall(sensorFile)
        return X, y
