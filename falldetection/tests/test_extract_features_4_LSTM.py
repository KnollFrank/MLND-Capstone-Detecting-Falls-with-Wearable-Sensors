from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.extract_features_4_LSTM import extract_features_4_LSTM


class ExtractFeatures4LSTMTestCase(TestCase):

    def test_extract_features_4_LSTM1(self):
        # GIVEN
        df = pd.DataFrame(
            {'Acc_X': [1.0],
             'Acc_Y': [4.0]})

        # WHEN
        features_actual = extract_features_4_LSTM(df, ['Acc_X', 'Acc_Y'])

        # THEN
        features_expected = np.array([[1.0, 4.0]])
        self.assertEquals(features_expected.tolist(), features_actual.tolist())

    def test_extract_features_4_LSTM2(self):
        # GIVEN
        df = pd.DataFrame(
            {'Acc_X': [1.0, 2.0, 3.0],
             'Acc_Y': [4.0, 5.0, 6.0]})

        # WHEN
        features_actual = extract_features_4_LSTM(df, ['Acc_Y', 'Acc_X'])

        # THEN
        features_expected = np.array([[4.0, 1.0],
                                      [5.0, 2.0],
                                      [6.0, 3.0]])
        self.assertEquals(features_expected.tolist(), features_actual.tolist())
