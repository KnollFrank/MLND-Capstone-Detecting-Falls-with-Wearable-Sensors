from unittest import TestCase

from falldetection.feature_extraction import *


class FeatureExtractionTestCase(TestCase):

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
             'Acc_Z': [1.0, 40.0, 5.0, 4.0, 5.0]},
            index=list('abcde'))
        index = get_index_of_maximum_total_acceleration(df)
        self.assertEquals(index, 'b')

    def test_get_window_around_maximum_total_acceleration(self):
        df = pd.DataFrame(
            {'Acc_X': [1.0, 20.0, 3.0, 4.0, 5.0],
             'Acc_Y': [1.0, 30.0, 4.0, 4.0, 5.0],
             'Acc_Z': [1.0, 40.0, 5.0, 4.0, 5.0]})
        df_sliced_actual = get_window_around_maximum_total_acceleration(df, half_window_size=1)
        df_sliced_expected = pd.DataFrame(
            {'Acc_X': [1.0, 20.0, 3.0],
             'Acc_Y': [1.0, 30.0, 4.0],
             'Acc_Z': [1.0, 40.0, 5.0]})
        self.assertTrue(df_sliced_expected.equals(df_sliced_actual))

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
        all_features_actual = extract_all_features(
            baseDir='../data/FallDataSet-Test',
            sensorFile='340535.txt',
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
                'feature': [
                    [1.0],
                    [2.0],
                    [3.0],
                    [4.0],
                    [5.0],
                    [6.0],
                    [7.0]]})
        self.assertTrue(all_features_expected.equals(all_features_actual))

    def test_extract_all_features2(self):
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 10000)
        pd.set_option('display.max_colwidth', 1000)
        all_features = extract_all_features(
            baseDir='../data/FallDataSet-Test',
            sensorFile='340535.txt')
        print("\n", all_features)
        print("dtypes:", all_features.dtypes)
        all_features.to_csv('../data/all_features.csv')

    def test_features2array(self):
        features = pd.DataFrame(
            {'Acc_X': [10.0, 11.0],
             'Mag_Z': [12.0, 13.0]})
        feature_array = features2array(features)
        self.assertEquals([10.0, 11.0, 12.0, 13.0], feature_array.tolist())
