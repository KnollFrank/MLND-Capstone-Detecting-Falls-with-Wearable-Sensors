import unittest as unittest
from unittest import TestCase

from falldetection.feature_extraction import *
from falldetection.sensor_files_provider import SensorFilesProvider


class FeatureExtractionTestCase(TestCase):

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
