from unittest import TestCase

import pandas as pd

from falldetection.feature_extraction import time_window


class FeatureExtractionTestCase(TestCase):

    def test_time_window(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_time_windowed = time_window(df, window_center_index=2, half_window_size=2)
        self.assertTrue(df.equals(df_time_windowed))

    def test_time_window2(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_time_windowed = time_window(df, window_center_index=2, half_window_size=1)
        self.assertTrue(pd.DataFrame({'Acc_X': [2.0, 3.0, 4.0]}).equals(df_time_windowed))
