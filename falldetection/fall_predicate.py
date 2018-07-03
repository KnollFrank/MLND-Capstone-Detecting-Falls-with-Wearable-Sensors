import re


def isFall(sensorFile):
    eight_or_nine = re.search('Testler Export/([89])', sensorFile).group(1)
    return eight_or_nine == '9'
