from unittest import TestCase

import pandas as pd

from falldetection.feature_extractor import FeatureExtractor


class FeatureExtractorTestCase(TestCase):

    def test_flatten_data_frame(self):
        df = pd.DataFrame(
            data={'Acc_X': [10.0, 11.0],
                  'Mag_Z': [12.0, 13.0]},
            index=['min', 'max'])
        df_flattened_actual = FeatureExtractor.flatten_data_frame(df)
        df_flattened_expected = pd.DataFrame(
            data={'Acc_X_min': [10.0],
                  'Acc_X_max': [11.0],
                  'Mag_Z_min': [12.0],
                  'Mag_Z_max': [13.0]})
        # TODO: DRY with similar print statements of actual and expected dataframes
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print('\ndf_flattened_expected:\n', df_flattened_expected)
        print('df_flattened_actual:\n', df_flattened_actual)
        self.assertTrue(df_flattened_expected.equals(df_flattened_actual))
