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
data = pd.read_csv('/home/brom/LAB/SLai_Folder/PeriodicRegression.model/energy_records_history.csv')
data = data[data['id']==150001]
data['energyKwh'].plot()
data['dt'] = data['timeFrom']
data['y'] = data['energyKwh']
data = data[['dt','y']]
data['dt'] = pd.to_datetime(data['dt'])

top_n = 5
# %% %%
PR = PeriodicRegression(top_n = top_n, max_correction = 300, cv=0.05, lags=[1,2,3,4,5,6,7], lag_freq='1D')
PR.fit(data)

PR.plot_train_results()
PR.plot_train_results(x_lim = ('2020-06-01','2020-08-09'),
                      y_lim = (400,1800))
PR.scores
dir(PR)
PR.plot_corrections()
#PR.plot_spectrum(log=True)
	MAE	MBE	RMSE	MAPE	r2_score
train	89.338889	-9.382801e-14	128.141860	10.987896	0.848517
test	96.992640	-1.005513e+01	131.613239	10.723795	0.868977
