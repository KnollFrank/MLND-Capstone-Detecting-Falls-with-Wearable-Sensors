from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.extract_time_series import extract_time_series


class ExtractTimeSeriesTestCase(TestCase):

    def test_extract_time_series1(self):
        self.__test_extract_time_series(
            df=pd.DataFrame({'Acc_X': [1.0], 'Acc_Y': [4.0]}),
            columns=['Acc_X', 'Acc_Y'],
            features_expected=np.array([[1.0, 4.0]]))

    def test_extract_time_series2(self):
        self.__test_extract_time_series(
            df=pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0], 'Acc_Y': [4.0, 5.0, 6.0]}),
            columns=['Acc_Y', 'Acc_X'],
            features_expected=np.array([[4.0, 1.0], [5.0, 2.0], [6.0, 3.0]]))

    def __test_extract_time_series(self, df, columns, features_expected):
        # WHEN
        features_actual = extract_time_series(df, columns)

        # THEN
        self.assertEquals(features_expected.tolist(), features_actual.tolist())
