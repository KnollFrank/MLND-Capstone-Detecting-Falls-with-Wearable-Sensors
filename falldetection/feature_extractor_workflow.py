import pandas as pd

from falldetection.fall_predicate import isFall
from falldetection.feature_extractor import FeatureExtractor
from falldetection.sensor_files_provider import SensorFilesProvider


class FeatureExtractorWorkflow:

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def extract_features(self, sensorFiles):
        sensorFiles_features_falls = [(sensorFile, self.feature_extractor(sensorFile), isFall(sensorFile)) for
                                      sensorFile in sensorFiles]
        return self.__asDataFrame(sensorFiles_features_falls)

    def __asDataFrame(self, sensorFiles_features_falls):
        sensorFiles, features, falls = zip(*sensorFiles_features_falls)
        return pd.DataFrame(
            {'sensorFile': sensorFiles,
             'fall': falls,
             'feature': features})


def extract_features_and_save():
    sensor_files = SensorFilesProvider(baseDir='../data/FallDataSet', sensorFile='340535.txt').provide_sensor_files()
    all_features = FeatureExtractorWorkflow(FeatureExtractor().extract_features).extract_features(sensor_files)
    all_features.to_csv('../data/all_features.csv')
