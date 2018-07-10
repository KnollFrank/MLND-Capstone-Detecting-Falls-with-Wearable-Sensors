from unittest import TestCase

import numpy as np
import pandas as pd

from falldetection.fall_predicate import isFall
from falldetection.time_series_extractor import TimeSeriesExtractor


class TimeSeriesExtractorTestCase(TestCase):

    def test_extract_time_series(self):
        # GIVEN
        sensorFile = '../data/FallDataSet/209/Testler Export/916/Test_1/340535.txt'
        df = pd.DataFrame({'Acc_X': [1.0, 2.0, 3.0], 'Acc_Y': [4.0, 5.0, 6.0]})
        time_series_extractor = TimeSeriesExtractor(
            sensor_file_2_df=lambda sensor_file: {sensorFile: df}[sensor_file],
            columns=['Acc_X', 'Acc_Y'])

        # WHEN
        X_actual, y_actual = time_series_extractor.extract_time_series(sensorFile)

        # THEN
        X_expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        y_expected = isFall(sensorFile)
        self.assertEquals(X_expected.tolist(), X_actual.tolist())
        self.assertEquals(y_expected, y_actual)
