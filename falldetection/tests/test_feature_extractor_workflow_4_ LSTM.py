import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.feature_extractor_4_LSTM import FeatureExtractor4LSTM
from falldetection.feature_extractor_workflow_4_LSTM import FeatureExtractorWorkflow4LSTM, extract_features_4_LSTM
from falldetection.sensor import Sensor
from falldetection.sensor_files_to_exclude import get_sensor_files_to_exclude_for


class FeatureExtractorWorkflow4LSTMTestCase(TestCase):

    def test_extract_features1(self):
        self.__test_extract_features(
            df_by_sensor_file={
                '../../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt':
                    pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0], 'Acc_Y': [4.0, 5.0, 6.0]}),

                '../../data/FallDataSet-Test/209/Testler Export/814/Test_6/340535.txt':
                    pd.DataFrame({'Acc_X': [10.0, 20.0, 30.0], 'Acc_Y': [40.0, 50.0, 60.0]})},
            columns=['Acc_X', 'Acc_Y'],
            X_expected=np.array([[[1.0, 4.0],
                                  [2.0, 5.0],
                                  [3.0, 6.0]],

                                 [[10.0, 40.0],
                                  [20.0, 50.0],
                                  [30.0, 60.0]]]),
            y_expected=np.array([[True], [False]]))

    def test_extract_features2(self):
        self.__test_extract_features(
            df_by_sensor_file={
                '../../data/FallDataSet-Test/209/Testler Export/814/Test_1/340535.txt':
                    pd.DataFrame({'Acc_X': [1.0], 'Mag_X': [4.0]}),

                '../../data/FallDataSet-Test/209/Testler Export/814/Test_6/340535.txt':
                    pd.DataFrame({'Acc_X': [10.0], 'Mag_X': [40.0]})},
            columns=['Mag_X', 'Acc_X'],
            X_expected=np.array([[[4.0, 1.0]], [[40.0, 10.0]]]),
            y_expected=np.array([[False], [False]]))

    def __test_extract_features(self, df_by_sensor_file, columns, X_expected, y_expected):
        # GIVEN
        feature_extractor_4_lstm = \
            FeatureExtractorWorkflow4LSTM(
                FeatureExtractor4LSTM(
                    sensor_file_2_df=df_by_sensor_file.get,
                    columns=columns))

        sensor_files = df_by_sensor_file.keys()

        # WHEN
        X_actual, y_actual = feature_extractor_4_lstm.extract_features(sensor_files)

        # THEN
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected.tolist(), y_actual.tolist())

    @unittest.SkipTest
    def test_extract_features_4_LSTM_RIGHT_THIGH(self):
        sensor = Sensor.RIGHT_THIGH
        features = extract_features_4_LSTM(
            sensor=sensor,
            baseDir='../../data/FallDataSet',
            sensor_files_to_exclude=get_sensor_files_to_exclude_for(sensor),
            columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
        print("features: ", features)

    @unittest.SkipTest
    def test_extract_features_4_LSTM_WAIST(self):
        sensor = Sensor.WAIST
        features = extract_features_4_LSTM(
            sensor=sensor,
            baseDir='../../data/FallDataSet',
            sensor_files_to_exclude=get_sensor_files_to_exclude_for(sensor),
            columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
        print("features: ", features)
