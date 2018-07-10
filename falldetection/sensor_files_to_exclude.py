from falldetection.sensor import Sensor


def get_sensor_files_to_exclude_for(sensor):
    return {Sensor.RIGHT_THIGH:
        [
            # IndexError: ../../data/FallDataSet/208/Testler Export/805/Test_1/340539.txt: not (0 <= -48 < 427 and 0 <= 53 <= 427)
            '208/Testler Export/805/Test_1/340539.txt',
            # IndexError: ../../data/FallDataSet/203/Testler Export/813/Test_1/340539.txt: not (0 <= -44 < 391 and 0 <= 57 <= 391)
            '203/Testler Export/813/Test_1/340539.txt',
            # IndexError: ../../data/FallDataSet/103/Testler Export/911/Test_5/340539.txt: not (0 <= 119 < 219 and 0 <= 220 <= 219)
            '103/Testler Export/911/Test_5/340539.txt',
            # TypeError: reduction operation 'argmax' not allowed for this dtype
            '109/Testler Export/901/Test_6/340539.txt',
            # IndexError: ../../data/FallDataSet/108/Testler Export/918/Test_5/340539.txt: not (0 <= 460 < 513 and 0 <= 561 <= 513)
            '108/Testler Export/918/Test_5/340539.txt',
            # IndexError: ../../data/FallDataSet/208/Testler Export/904/Test_6/340539.txt: not (0 <= 193 < 291 and 0 <= 294 <= 291)
            '208/Testler Export/904/Test_6/340539.txt',
            # IndexError: ../../data/FallDataSet/207/Testler Export/904/Test_4/340539.txt: not (0 <= 146 < 223 and 0 <= 247 <= 223)
            '207/Testler Export/904/Test_4/340539.txt'
        ]}[sensor]
