import pandas as pd


def read_sensor_file(sensorFile):
    df = pd.read_csv(sensorFile,
                     skiprows=4,
                     sep='\t',
                     usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])
    return df.dropna()
