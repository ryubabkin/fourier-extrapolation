import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import functions
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
# %% %%

data = pd.read_csv('./Weather.csv', parse_dates=True)
data = data[data['location']=='san severo'].reset_index(drop=True)

data['dt'] = data['time']
data['dt'] = pd.to_datetime(data['dt'])
data = data.sort_values('dt').reset_index(drop=True)
data['y'] = data['solarIrradiance']
data = data[['dt','y']]

# %% %
plt.figure(figsize=(15,7))
plt.plot(data['dt'],data['y'])
# %% %
data['s1'] = data['y'].shift(24)
data['s2'] = data['y'].shift(48)
data['s3'] = data['y'].shift(72)
data = data.dropna()
LR = LinearRegression()

LR.fit(data.drop(['dt','y'],axis=1)[:-3000],data['y'][:-3000])
P = LR.predict(data.drop(['dt','y'],axis=1)[-3000:])
# %% %
plt.figure(figsize=(15,7))
plt.plot(data['dt'],data['y'])
plt.plot(data['dt'][-3000:],P, alpha = 0.7)
plt.xlim(pd.to_datetime('2020-04-01'), pd.to_datetime('2020-07-09'))
# %% %%
functions.MAE(data['y'][-3000:],P)
functions.MAPE(data['y'][-3000:],P)
functions.MBE(data['y'][-3000:],P)








# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Periodic Features
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
import functions
from PeriodicRegressionClass import PeriodicRegression

PR = PeriodicRegression(time_step='1h',  # time series step
                        top_n=5,  # top n frequencies
                        n_max=20,  # max frequencies for top_n = "auto"
                        cv=0.1,  # validation part
                        lags=[1,2,3,4,5],  # list of lag steps
                        lag_freq='1d',  # time  step for lags
                        max_correction=300,  # max length for correction, 0.1 of length
                        fill_ranges=['3h','3d'])
PR.fit(data)
PR.regressor
PR.__dir__()
PR.plot_train_results()
PR.plot_train_results(x_lim=('2020-04-01', '2020-07-09'))#, y_lim=(800, 1600))
PR.plot_spectrum(log=False)

PR.scores
PR.plot_missing_data(frame=24)

PR._utils
