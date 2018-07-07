from unittest import TestCase

from falldetection.sensor_file_reader import read_sensor_file


class ReadSensorFileTestCase(TestCase):

    def test_read_sensor_file(self):
        df = read_sensor_file('340535-havingNaN.txt')
        self.assertEquals(self.__number_of_NaNs(df), 0)

    def __number_of_NaNs(self, df):
        return df.isnull().sum().sum()
