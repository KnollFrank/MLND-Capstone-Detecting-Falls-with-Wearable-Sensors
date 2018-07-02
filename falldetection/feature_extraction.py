import pandas as pd
from IPython.display import display

# %matplotlib inline

data = pd.read_csv("../data/FallDataSet/101/Testler Export/901/Test_1/340535.txt", skiprows=4, sep='\t',
                   usecols=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])

# Success - Display the first record
display(data.head())
data.describe()
