from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.feature_extractor_workflow_4_LSTM import FeatureExtractorWorkflow4LSTM


class FeatureExtractorWorkflow4LSTMTestCase(TestCase):

    # TODO: DRY with test_extract_features2
    def test_extract_features1(self):
        # GIVEN
        sensor_files = ['../../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt',
                        '../../data/FallDataSet-Test/209/Testler Export/814/Test_6/340535.txt']

        feature_extractor_4_lstm = FeatureExtractorWorkflow4LSTM(
            feature_extractor=lambda sensorFile: {
                sensor_files[0]:
                    pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0], 'Acc_Y': [4.0, 5.0, 6.0]}),

                sensor_files[1]:
                    pd.DataFrame({'Acc_X': [10.0, 20.0, 30.0], 'Acc_Y': [40.0, 50.0, 60.0]})}
            [sensorFile],
            columns=['Acc_X', 'Acc_Y'])

        # WHEN
        X_actual, y_actual = feature_extractor_4_lstm.extract_features(sensor_files)

        # THEN
        X_expected = np.array([[[1.0, 4.0],
                                [2.0, 5.0],
                                [3.0, 6.0]],

                               [[10.0, 40.0],
                                [20.0, 50.0],
                                [30.0, 60.0]]])
        y_expected = np.array([[True], [False]])
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected.tolist(), y_actual.tolist())

    def test_extract_features2(self):
        # GIVEN
        sensor_files = ['../../data/FallDataSet-Test/209/Testler Export/814/Test_1/340535.txt',
                        '../../data/FallDataSet-Test/209/Testler Export/814/Test_6/340535.txt']

        feature_extractor_4_lstm = FeatureExtractorWorkflow4LSTM(
            feature_extractor=lambda sensorFile: {
                sensor_files[0]:
                    pd.DataFrame({'Acc_X': [1.0], 'Mag_X': [4.0]}),

                sensor_files[1]:
                    pd.DataFrame({'Acc_X': [10.0], 'Mag_X': [40.0]})}
            [sensorFile],
            columns=['Mag_X', 'Acc_X'])

        # WHEN
        X_actual, y_actual = feature_extractor_4_lstm.extract_features(sensor_files)

        # THEN
        X_expected = np.array([[[4.0, 1.0]], [[40.0, 10.0]]])
        y_expected = np.array([[False], [False]])
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected.tolist(), y_actual.tolist())
