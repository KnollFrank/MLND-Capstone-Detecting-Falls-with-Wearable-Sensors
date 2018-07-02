from unittest import TestCase

import pandas as pd

from falldetection.feature_extraction import *


class FeatureExtractionTestCase(TestCase):

    def test_time_window1(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_time_windowed = time_window(df, window_center_index=2, half_window_size=2)
        self.assertTrue(df.equals(df_time_windowed))

    def test_time_window2(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_time_windowed = time_window(df, window_center_index=2, half_window_size=1)
        self.assertTrue(pd.DataFrame({'Acc_X': [2.0, 3.0, 4.0]}).equals(df_time_windowed))

    def test_time_window3(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        with self.assertRaises(IndexError):
            time_window(df, window_center_index=2, half_window_size=3)

    def test_get_index_of_maximum_total_acceleration1(self):
        df = pd.DataFrame(
            {'Acc_X': [1.0, 2.0, 30.0, 4.0, 5.0],
             'Acc_Y': [1.0, 2.0, 40.0, 4.0, 5.0],
             'Acc_Z': [1.0, 2.0, 50.0, 4.0, 5.0]})
        index = get_index_of_maximum_total_acceleration(df)
        self.assertEquals(index, 2)

    def test_get_index_of_maximum_total_acceleration2(self):
        df = pd.DataFrame(
            {'Acc_X': [1.0, 20.0, 3.0, 4.0, 5.0],
             'Acc_Y': [1.0, 30.0, 4.0, 4.0, 5.0],
             'Acc_Z': [1.0, 40.0, 5.0, 4.0, 5.0]})
        index = get_index_of_maximum_total_acceleration(df)
        self.assertEquals(index, 1)
