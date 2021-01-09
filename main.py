import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

import functions
from PeriodicRegressionClass import PeriodicRegression
import warnings
warnings.filterwarnings('ignore')
# %% %%
data = pd.read_csv('/home/brom/LAB/SLai_Folder/FERegression.model/energy_records_history.csv')
data = data[data['id']==150001]
data['energyKwh'].plot()
data['dt'] = data['timeFrom']
data['y'] = data['energyKwh']
data = data[['dt','y']]
data['dt'] = pd.to_datetime(data['dt'])
# %% %%
data = pd.DataFrame()
data['dt'] = pd.date_range('2020-01-01 00:00:00', periods=3000, freq='1d')
X = np.arange(0,len(data))
A1, A2, A3 = 10, 0, 20
f1, f2, f3 = 0.005, 0.05, 0.002
p1, p2, p3 = 0.4,0,0.3

data['y'] = A1*np.sin(2*np.pi*X*f1+p1)+A2*np.cos(2*np.pi*X*f2+p2)+A3*np.sin(2*np.pi*X*f3+p3)+np.random.normal(0,1,len(data))*5
data['y'] = abs(data['y'])
data['y'].plot()
data.shape
# %% %%
# %% %%
top_n = 6
L = 10000
correction, result = functions.find_length_correction(data[:-L],1000, 6)
result['abs'].plot()
correction
top_n = 10
PR = PeriodicRegression()
PR.fit(data[:-L-correction], top_n = top_n, cv=0.1)
restored = PR.predict(np.arange(0,len(data)))

plt.figure(figsize=(10,7))
plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(np.arange(0,len(data)),restored, alpha = 0.75)
plt.axvline(len(data)-L,c='r')
#plt.xlim(18000,18500)
plt.show()

PR.plot_spectrum(log=True)
# %% %%
L
