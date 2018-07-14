import pandas as pd

from falldetection.fall_predicate import isFall
from falldetection.feature_extractor import FeatureExtractor
from falldetection.sensor_files_provider import SensorFilesProvider
from falldetection.sensor_files_to_exclude import get_sensor_files_to_exclude_for


class FeatureExtractorWorkflow:

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def extract_features(self, sensorFiles):
        return pd.concat(self.__create_dataFrames(sensorFiles), ignore_index=True, axis='index')

    def __create_dataFrames(self, sensorFiles):
        return [self.__create_dataFrame(sensorFile) for sensorFile in sensorFiles]

    def __create_dataFrame(self, sensorFile):
        return pd.concat((self.sensorFile_and_fall_df(sensorFile), self.feature_extractor(sensorFile)), axis='columns')

    def sensorFile_and_fall_df(self, sensorFile):
        return pd.DataFrame(data={'sensorFile': [sensorFile], 'fall': isFall(sensorFile)})


def extract_features_and_save(sensor, baseDir, csv_file, autocovar_num, dft_amplitudes_num):
    def extract_features():
        def get_sensor_files():
            return createSensorFilesProvider().provide_sensor_files()

        def createSensorFilesProvider():
            return SensorFilesProvider(baseDir, sensor, get_sensor_files_to_exclude_for(sensor))

        def createFeatureExtractorWorkflow():
            return FeatureExtractorWorkflow(create_feature_extractor())

        def create_feature_extractor():
            return FeatureExtractor(autocovar_num, dft_amplitudes_num).extract_features

        return createFeatureExtractorWorkflow().extract_features(get_sensor_files())

    features = extract_features()
    features.to_csv(csv_file)

# TODO: als Benchmark zusätzlich zu den im proposal definierten noch einen Schwellwert-Algorithmus z.B. für die totale Beschleunigung implementieren oder besser einen machine learning Algorithmus (z.B. SVM), der nur die totale Beschleunigung verwendet.
