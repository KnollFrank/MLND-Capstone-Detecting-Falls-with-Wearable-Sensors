import os

from falldetection.sensor import Sensor


class SensorFilesProvider:

    def __init__(self, baseDir, sensor, sensor_files_to_exclude=()):
        self.baseDir = baseDir
        self.sensor = sensor
        self.sensor_files_to_exclude = [baseDir + '/' + sensor_file_to_exclude for sensor_file_to_exclude in
                                        sensor_files_to_exclude]

    def provide_sensor_files(self):
        return [sensor_file for sensor_file in
                self.__provide_sensor_files()
                if not self.__shall_exclude(sensor_file)]

    def __provide_sensor_files(self):
        for root, dirs, files in os.walk(self.baseDir):
            for file in files:
                if Sensor.from_file(file) is self.sensor:
                    yield os.path.join(root, file)

    def __shall_exclude(self, sensor_file):
        return sensor_file in self.sensor_files_to_exclude or "Fail" in sensor_file
