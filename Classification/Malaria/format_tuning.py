import os

import numpy as np
import pandas as pd

# /media/yaniv/data2/datasets/Malaria/VGG__2019-06-25_0137/results-0_071.csv
root = '/media/yaniv/data2/datasets/Malaria'
folder = 'VGG__2019-06-25_0137'
file = 'results-0_071'
df = pd.read_csv(os.path.join(root, folder, file + '.csv'))

thr = 0.99
val = df['infected'].to_numpy()
val = np.where(val >= thr, np.ones_like(val), val)
val = np.where(val <= 1 - thr, np.zeros_like(val), val)
df['infected'] = val

df.to_csv(os.path.join(root, folder, file + '_x' + '.csv'))

