from falldetection.sensor import Sensor


def get_sensor_files_to_exclude_for(sensor):
    return {
        Sensor.RIGHT_THIGH: [
            # IndexError: not (0 <= -48 < 427 and 0 <= 53 <= 427)
            '208/Testler Export/805/Test_1/340539.txt',

            # IndexError: not (0 <= -44 < 391 and 0 <= 57 <= 391)
            '203/Testler Export/813/Test_1/340539.txt',

            # IndexError: not (0 <= 119 < 219 and 0 <= 220 <= 219)
            '103/Testler Export/911/Test_5/340539.txt',

            # TypeError: reduction operation 'argmax' not allowed for this dtype
            '109/Testler Export/901/Test_6/340539.txt',

            # IndexError: not (0 <= 460 < 513 and 0 <= 561 <= 513)
            '108/Testler Export/918/Test_5/340539.txt',

            # IndexError: not (0 <= 193 < 291 and 0 <= 294 <= 291)
            '208/Testler Export/904/Test_6/340539.txt',

            # IndexError: not (0 <= 146 < 223 and 0 <= 247 <= 223)
            '207/Testler Export/904/Test_4/340539.txt'
        ],

        Sensor.WAIST: [
            # IndexError: not (0 <= 401 < 455 and 0 <= 502 <= 455)
            '209/Testler Export/919/Test_5/340535.txt',

            # IndexError: not (0 <= -44 < 391 and 0 <= 57 <= 391)
            '203/Testler Export/813/Test_1/340535.txt',

            # IndexError: not (0 <= 302 < 388 and 0 <= 403 <= 388)
            '207/Testler Export/917/Test_1/340535.txt',

            # TypeError: reduction operation 'argmax' not allowed for this dtype
            '109/Testler Export/901/Test_6/340535.txt',

            # IndexError: not (0 <= 223 < 255 and 0 <= 324 <= 255)
            '208/Testler Export/917/Test_5/340535.txt',

            # IndexError: not (0 <= 275 < 339 and 0 <= 376 <= 339)
            '103/Testler Export/917/Test_5/340535.txt',

            # IndexError: not (0 <= 275 < 339 and 0 <= 376 <= 339)
            '103/Testler Export/917/Test_4/340535.txt',

            # IndexError: not (0 <= 266 < 199 and 0 <= 367 <= 199)
            '205/Testler Export/917/Test_5/340535.txt',

            # IndexError: not (0 <= 254 < 254 and 0 <= 355 <= 254)
            '107/Testler Export/917/Test_3/340535.txt'
        ]}[sensor]
