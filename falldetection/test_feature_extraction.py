import unittest as unittest
from unittest import TestCase

from falldetection.feature_extraction import *
from falldetection.sensor_files_provider import SensorFilesProvider


class FeatureExtractionTestCase(TestCase):

    def test_extract_features1(self):
        df = pd.DataFrame(
            {'Acc_X': [1.0, 2.0, 3.0],
             'Gyr_X': [4.0, 5.0, 6.0]})
        features_actual = extract_features(df)
        print(features_actual)

        features_expected = pd.DataFrame(
            index=['min', 'max', 'mean', 'var', 'skew', 'kurtosis'],
            columns=['Acc_X', 'Gyr_X'],
            dtype=np.float64)
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
        df = pd.read_csv(
            "../data/FallDataSet/101/Testler Export/901/Test_1/340535.txt",
            skiprows=4,
            sep='\t',
            usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
        features = extract_features(df)
        print("\n", features)

    def test_extract_all_features1(self):
        sensor_files_provider = SensorFilesProvider(baseDir='../data/FallDataSet-Test', sensorFile='340535.txt')
        all_features_actual = extract_all_features(
            sensorFiles=sensor_files_provider.provide_sensor_files(),
            feature_extractor=lambda sensorFile: {
                '../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt': [1.0],
                '../data/FallDataSet-Test/209/Testler Export/914/Test_6/340535.txt': [2.0],
                '../data/FallDataSet-Test/209/Testler Export/801/Test_1/340535.txt': [3.0],
                '../data/FallDataSet-Test/209/Testler Export/801/Test_2/340535.txt': [4.0],
                '../data/FallDataSet-Test/101/Testler Export/801/Test_1/340535.txt': [5.0],
                '../data/FallDataSet-Test/101/Testler Export/801/Test_2/340535.txt': [6.0],
                '../data/FallDataSet-Test/101/Testler Export/920/Test_1/340535.txt': [7.0]}[sensorFile])
        all_features_expected = pd.DataFrame(
            {'sensorFile': [
                '../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt',
                '../data/FallDataSet-Test/209/Testler Export/914/Test_6/340535.txt',
                '../data/FallDataSet-Test/209/Testler Export/801/Test_1/340535.txt',
                '../data/FallDataSet-Test/209/Testler Export/801/Test_2/340535.txt',
                '../data/FallDataSet-Test/101/Testler Export/801/Test_1/340535.txt',
                '../data/FallDataSet-Test/101/Testler Export/801/Test_2/340535.txt',
                '../data/FallDataSet-Test/101/Testler Export/920/Test_1/340535.txt'],
                'fall': [True, True, False, False, False, False, True],
                'feature': [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0]]})
        self.assertTrue(all_features_expected.equals(all_features_actual))

    @unittest.SkipTest
    def test_extract_all_features2(self):
        extract_all_features_and_save()

    def test_features2array(self):
        features = pd.DataFrame(
            {'Acc_X': [10.0, 11.0],
             'Mag_Z': [12.0, 13.0]})
        feature_array = features2array(features)
        self.assertEquals([10.0, 11.0, 12.0, 13.0], feature_array.tolist())

    def test_is_fall1(self):
        fall = isFall('../data/FallDataSet/209/Testler Export/916/Test_1/340535.txt')
        self.assertEquals(fall, True)

    def test_is_fall2(self):
        fall = isFall('../data/FallDataSet/209/Testler Export/813/Test_6/340535.txt')
        self.assertEquals(fall, False)
