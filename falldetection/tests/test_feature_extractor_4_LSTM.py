from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.fall_predicate import isFall
from falldetection.feature_extractor_4_LSTM import FeatureExtractor4LSTM


class FeatureExtractor4LSTMTestCase(TestCase):

    def test_extract_features(self):
        # GIVEN
        sensorFile = '../data/FallDataSet/209/Testler Export/916/Test_1/340535.txt'
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0], 'Acc_Y': [4.0, 5.0, 6.0]})
        feature_extractor_4_lstm = FeatureExtractor4LSTM(lambda sensor_file: {sensorFile: df}[sensor_file],
                                                         ['Acc_X', 'Acc_Y'])

        # WHEN
        X_actual, y_actual = feature_extractor_4_lstm.extract_features(sensorFile)

        # THEN
        X_expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        y_expected = isFall(sensorFile)
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected, y_actual)
