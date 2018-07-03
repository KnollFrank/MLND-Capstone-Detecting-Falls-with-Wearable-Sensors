from unittest import TestCase

import pandas as pd

from falldetection.extract_features import extract_features
from falldetection.sensor_file_reader import read_sensor_file


class ExtractFeaturesTestCase(TestCase):

    def test_extract_features1(self):
        df = pd.DataFrame(
            {'Acc_X': [1.0, 2.0, 3.0],
             'Gyr_X': [4.0, 5.0, 6.0]})
        features_actual = extract_features(df)
        print(features_actual)

        features_expected = pd.DataFrame(
            index=['min', 'max', 'mean', 'var', 'skew', 'kurtosis'],
            columns=['Acc_X', 'Gyr_X'],
            dtype=pd.np.float64)
        features_expected.at['min', 'Acc_X'] = 1.0
        features_expected.at['min', 'Gyr_X'] = 4.0

        features_expected.at['max', 'Acc_X'] = 3.0
        features_expected.at['max', 'Gyr_X'] = 6.0

        mean_Acc_X = (1.0 + 2.0 + 3.0) / 3
        features_expected.at['mean', 'Acc_X'] = mean_Acc_X
        mean_Gyr_X = (4.0 + 5.0 + 6.0) / 3
        features_expected.at['mean', 'Gyr_X'] = mean_Gyr_X

        features_expected.at['var', 'Acc_X'] = ((1.0 - mean_Acc_X) ** 2 + (2.0 - mean_Acc_X) ** 2 + (
                3.0 - mean_Acc_X) ** 2) / 3
        features_expected.at['var', 'Gyr_X'] = ((4.0 - mean_Gyr_X) ** 2 + (5.0 - mean_Gyr_X) ** 2 + (
                6.0 - mean_Gyr_X) ** 2) / 3

        features_expected.at['skew', 'Acc_X'] = ((1.0 - mean_Acc_X) ** 3 + (2.0 - mean_Acc_X) ** 3 + (
                3.0 - mean_Acc_X) ** 3) / (3 * features_expected.at['var', 'Acc_X'] ** 3)
        features_expected.at['skew', 'Gyr_X'] = ((4.0 - mean_Gyr_X) ** 3 + (5.0 - mean_Gyr_X) ** 3 + (
                6.0 - mean_Gyr_X) ** 3) / (3 * features_expected.at['var', 'Gyr_X'] ** 3)

        features_expected.loc['kurtosis', :] = df.kurtosis()[features_expected.columns].values

        self.assertTrue(features_expected.equals(features_actual))

    def test_extract_features2(self):
        df = read_sensor_file("../data/FallDataSet/101/Testler Export/901/Test_1/340535.txt")
        features = extract_features(df)
        print("\n", features)
