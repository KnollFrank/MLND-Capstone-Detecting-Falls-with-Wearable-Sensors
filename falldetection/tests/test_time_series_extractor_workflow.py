import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.sensor import Sensor
from falldetection.time_series_extractor import TimeSeriesExtractor
from falldetection.time_series_extractor_workflow import TimeSeriesExtractorWorkflow, extract_time_series


class TimeSeriesExtractorWorkflowTestCase(TestCase):

    def test_extract_time_series1(self):
        self.__test_extract_time_series(
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

    def test_extract_time_series2(self):
        self.__test_extract_time_series(
            df_by_sensor_file={
                '../../data/FallDataSet-Test/209/Testler Export/814/Test_1/340535.txt':
                    pd.DataFrame({'Acc_X': [1.0], 'Mag_X': [4.0]}),

                '../../data/FallDataSet-Test/209/Testler Export/814/Test_6/340535.txt':
                    pd.DataFrame({'Acc_X': [10.0], 'Mag_X': [40.0]})},
            columns=['Mag_X', 'Acc_X'],
            X_expected=np.array([[[4.0, 1.0]], [[40.0, 10.0]]]),
            y_expected=np.array([[False], [False]]))

    def __test_extract_time_series(self, df_by_sensor_file, columns, X_expected, y_expected):
        # GIVEN
        time_series_extractor_workflow = \
            TimeSeriesExtractorWorkflow(
                TimeSeriesExtractor(
                    sensor_file_2_df=df_by_sensor_file.get,
                    columns=columns))

        sensor_files = df_by_sensor_file.keys()

        # WHEN
        X_actual, y_actual = time_series_extractor_workflow.extract_time_series(sensor_files)

        # THEN
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected.tolist(), y_actual.tolist())

    @unittest.SkipTest
    def test_extract_time_series_RIGHT_THIGH(self):
        features = extract_time_series(
            sensor=Sensor.RIGHT_THIGH,
            baseDir='../../data/FallDataSet',
            columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
        print("features: ", features)

    @unittest.SkipTest
    def test_extract_time_series_WAIST(self):
        features = extract_time_series(
            sensor=Sensor.WAIST,
            baseDir='../../data/FallDataSet',
            columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
        print("features: ", features)
