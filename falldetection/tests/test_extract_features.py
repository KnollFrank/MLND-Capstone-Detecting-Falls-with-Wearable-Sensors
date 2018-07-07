from unittest import TestCase

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf

from falldetection.extract_features import extract_features


# TODO: alle features (DFT) implementieren.


class ExtractFeaturesTestCase(TestCase):

    def __test_extract_features(self, autocorr_num, dft_amplitudes_num, add_expected_autocorr_of_df_2_features,
                                add_expected_dft_amplitudes_of_df_2_features):
        # build
        df = pd.DataFrame(
            {'Acc_X': [1.0, 2.0, 3.0],
             'Gyr_X': [4.0, 5.0, 6.0]})

        # execute
        features_actual = extract_features(df, autocorr_num, dft_amplitudes_num)

        # test
        print("features_actual:\n", features_actual)

        features_expected = pd.DataFrame(columns=['Acc_X', 'Gyr_X'], dtype=pd.np.float64)
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

        add_expected_autocorr_of_df_2_features(df=df, features=features_expected)

        add_expected_dft_amplitudes_of_df_2_features(df=df, features=features_expected)
        print("features_expected:\n", features_expected)

        self.assertTrue(features_expected.equals(features_actual))

    def test_extract_features1(self):
        def add_expected_autocorr_of_df_2_features(df, features):
            features.at['autocorr_lag_1', 'Acc_X'] = acovf(df['Acc_X'])[1]
            features.at['autocorr_lag_1', 'Gyr_X'] = acovf(df['Gyr_X'])[1]

        def add_expected_dft_amplitudes_of_df_2_features(df, features):
            features.at['dft_amplitude_1', 'Acc_X'] = np.abs(np.fft.fft(df['Acc_X'])[0])
            features.at['dft_amplitude_1', 'Gyr_X'] = np.abs(np.fft.fft(df['Gyr_X'])[0])

        self.__test_extract_features(
            autocorr_num=1,
            dft_amplitudes_num=1,
            add_expected_autocorr_of_df_2_features=add_expected_autocorr_of_df_2_features,
            add_expected_dft_amplitudes_of_df_2_features=add_expected_dft_amplitudes_of_df_2_features)

    def test_extract_features2(self):
        def add_expected_autocorr_of_df_2_features(df, features):
            features.at['autocorr_lag_1', 'Acc_X'] = acovf(df['Acc_X'])[1]
            features.at['autocorr_lag_2', 'Acc_X'] = acovf(df['Acc_X'])[2]
            features.at['autocorr_lag_1', 'Gyr_X'] = acovf(df['Gyr_X'])[1]
            features.at['autocorr_lag_2', 'Gyr_X'] = acovf(df['Gyr_X'])[2]

        def add_expected_dft_amplitudes_of_df_2_features(df, features):
            features.at['dft_amplitude_1', 'Acc_X'] = np.abs(np.fft.fft(df['Acc_X'])[0])
            features.at['dft_amplitude_2', 'Acc_X'] = np.abs(np.fft.fft(df['Acc_X'])[1])
            features.at['dft_amplitude_1', 'Gyr_X'] = np.abs(np.fft.fft(df['Gyr_X'])[0])
            features.at['dft_amplitude_2', 'Gyr_X'] = np.abs(np.fft.fft(df['Gyr_X'])[1])

        self.__test_extract_features(
            autocorr_num=2,
            dft_amplitudes_num=2,
            add_expected_autocorr_of_df_2_features=add_expected_autocorr_of_df_2_features,
            add_expected_dft_amplitudes_of_df_2_features=add_expected_dft_amplitudes_of_df_2_features)
