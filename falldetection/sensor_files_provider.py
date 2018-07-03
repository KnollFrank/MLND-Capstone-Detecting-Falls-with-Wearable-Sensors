import os


class SensorFilesProvider:

    def __init__(self, baseDir, sensorFile):
        self.baseDir = baseDir
        self.sensorFile = sensorFile

    def provide_sensor_files(self):
        return [sensor_file for sensor_file in
                self.__provide_sensor_files()
                if not self.__shall_exclude(sensor_file)]

    def __provide_sensor_files(self):
        for root, dirs, files in os.walk(self.baseDir):
            for file in files:
                if file == self.sensorFile:
                    yield os.path.join(root, file)

    def __shall_exclude(self, sensor_file):
        excluded_sensor_files = \
            (self.baseDir + '/209/Testler Export/919/Test_5/340535.txt',
             self.baseDir + '/203/Testler Export/813/Test_1/340535.txt',
             self.baseDir + '/207/Testler Export/917/Test_1/340535.txt',
             self.baseDir + '/109/Testler Export/901/Test_6/340535.txt')
        return sensor_file in excluded_sensor_files or "Fail" in sensor_file
