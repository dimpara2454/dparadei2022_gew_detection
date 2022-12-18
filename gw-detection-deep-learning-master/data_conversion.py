import h5py
import numpy as np
import pandas as pd

filename = "/media/fastdata/paranous/gwdata/detection/dataset-4/v2/train_background_s24w61w_1.hdf"

with h5py.File(filename, "r") as f:
    h1 = f.get('H1')
    col1 = np.array(h1.get('1238205077'))

with h5py.File(filename, "r") as f:
    h1 = f.get('L1')
    col2 = np.array(h1.get('1238205077'))

col1 = np.reshape(col1, (1, -1))
col2 = np.reshape(col2, (1, -1))
X = np.concatenate((col1, col2), axis=0)
Y = np.random.randint(0, 2, (X.shape[0], 1))
data = np.concatenate((X, Y), axis=1)
dataframe = pd.DataFrame(data, columns=['H1', 'L1', 'OT'])

dataframe.to_csv('DATA.csv')
