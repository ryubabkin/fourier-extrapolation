import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import r2_score

import functions
from PeriodicRegressionClass import PeriodicRegression
import warnings
warnings.filterwarnings('ignore')
# %% %%

data = pd.read_csv('/Users/wamiwamisoftware/Desktop/Work/LAB/AREAM/Data/fsight_weather_records_local_CLEAR.csv', parse_dates=True)
data['location'].unique()
data = data[data['location']=='lucija'].reset_index(drop=True)

data['dt'] = data['time']
data['y'] = data['tempC']
data = data[['dt','y']]
data['dt'] = pd.to_datetime(data['dt'])
data = data.sort_values('dt').reset_index(drop=True)

data['y'].plot()

# %% %%
top_n=5
PR = PeriodicRegression(top_n=top_n, max_correction = 300, cv=0.2, lags=[2,3,4,5,6,7], lag_freq='1d')
PR.fit(data)
PR.plot_train_results()
PR.plot_train_results(x_lim=('2020-01-01', '2020-07-09'))#, y_lim=(800, 1600))
PR.plot_spectrum(log=True)
PR.__dir__()
PR.scores
PR._utils.missing
PR.plot_missing_data(frame=96)
