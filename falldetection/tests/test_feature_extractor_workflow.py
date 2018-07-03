import unittest as unittest
from unittest import TestCase

from falldetection.feature_extractor_workflow import *
from falldetection.sensor_files_provider import SensorFilesProvider


class FeatureExtractionTestCase(TestCase):

    def test_extract_all_features1(self):
        # build
        feature_extractor = FeatureExtractorWorkflow(lambda sensorFile: {
            '../../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt': [1.0],
            '../../data/FallDataSet-Test/209/Testler Export/914/Test_6/340535.txt': [2.0],
            '../../data/FallDataSet-Test/209/Testler Export/801/Test_1/340535.txt': [3.0],
            '../../data/FallDataSet-Test/209/Testler Export/801/Test_2/340535.txt': [4.0],
            '../../data/FallDataSet-Test/101/Testler Export/801/Test_1/340535.txt': [5.0],
            '../../data/FallDataSet-Test/101/Testler Export/801/Test_2/340535.txt': [6.0],
            '../../data/FallDataSet-Test/101/Testler Export/920/Test_1/340535.txt': [7.0]}[sensorFile])

        sensor_files = SensorFilesProvider(baseDir='../../data/FallDataSet-Test',
                                           sensorFile='340535.txt').provide_sensor_files()
        features_expected = pd.DataFrame(
            {'sensorFile': [
                '../../data/FallDataSet-Test/209/Testler Export/914/Test_1/340535.txt',
                '../../data/FallDataSet-Test/209/Testler Export/914/Test_6/340535.txt',
                '../../data/FallDataSet-Test/209/Testler Export/801/Test_1/340535.txt',
                '../../data/FallDataSet-Test/209/Testler Export/801/Test_2/340535.txt',
                '../../data/FallDataSet-Test/101/Testler Export/801/Test_1/340535.txt',
                '../../data/FallDataSet-Test/101/Testler Export/801/Test_2/340535.txt',
                '../../data/FallDataSet-Test/101/Testler Export/920/Test_1/340535.txt'],
                'fall': [True, True, False, False, False, False, True],
                'feature': [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0]]})

        # execute
        features_actual = feature_extractor.extract_features(sensor_files)

        # test
        self.assertTrue(features_expected.equals(features_actual))

    @unittest.SkipTest
    def test_extract_all_features2(self):
        extract_features_and_save()
