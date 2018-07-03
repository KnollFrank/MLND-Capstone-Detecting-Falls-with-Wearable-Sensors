from unittest import TestCase

import pandas as pd

from falldetection.feature_extractor import FeatureExtractor


class FeatureExtractorTestCase(TestCase):

    def test_features2array(self):
        features = pd.DataFrame(
            {'Acc_X': [10.0, 11.0],
             'Mag_Z': [12.0, 13.0]})
        feature_array = FeatureExtractor.features2array(features)
        self.assertEquals([10.0, 11.0, 12.0, 13.0], feature_array.tolist())
