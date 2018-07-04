from enum import Enum


class Sensor(Enum):
    HEAD = '340506.txt'
    CHEST = '340527.txt'
    WAIST = '340535.txt'
    RIGHT_WRIST = '340537.txt'
    RIGHT_THIGH = '340539.txt'
    RIGHT_ANKLE = '340540.txt'

    @staticmethod
    def from_file(file):
        for sensor in Sensor:
            if sensor.value == file:
                return sensor

        return None
