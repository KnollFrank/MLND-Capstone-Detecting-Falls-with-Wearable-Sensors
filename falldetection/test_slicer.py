from unittest import TestCase

import pandas as pd

from falldetection.slicer import slice_with_window


class SlicerTestCase(TestCase):

    def test_slice_with_window1(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_sliced = slice_with_window(df, window_center_index=2, half_window_size=2)
        self.assertTrue(df.equals(df_sliced))

    def test_slice_with_window2(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_sliced_actual = slice_with_window(df, window_center_index=2, half_window_size=1)
        df_sliced_expected = pd.DataFrame({'Acc_X': [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        self.assertTrue(df_sliced_expected.equals(df_sliced_actual))

    def test_slice_with_window3(self):
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0, 4.0, 5.0]})
        with self.assertRaises(IndexError):
            slice_with_window(df, window_center_index=2, half_window_size=3)
