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
def MAE(E,F):
    return np.mean(abs(E-F))
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
L = 32000
correction, result = functions.find_length_correction(data[:L],1000, 6)
PR = PeriodicRegression()
PR.fit(data[:L-correction], top_n = top_n, cv=0.1)
pr_predict = PR.predict(np.arange(0,len(data)))
FREQS = PR.spectrum[PR.spectrum['peak']==1].sort_values('abs').tail(top_n)['freq']
DF = pd.DataFrame()
DF['dt'] = data['dt']
DF['y'] = data['y']
X = np.arange(len(DF['dt']))
i = 1
for freq in FREQS:
    DF['freq'+str(i)+'_s'] = np.sin(2*np.pi*freq*X)
    DF['freq'+str(i)+'_c'] = np.cos(2*np.pi*freq*X)
    i+=1

regr = LR()
regr.fit(DF.drop(['dt','y'],axis=1)[:L],DF['y'][:L])
P = regr.predict(DF.drop(['dt','y'],axis=1))



regr1 = LR()
regr1.fit(pr_predict[:L].reshape(-1, 1),DF['y'][:L])
P1 = regr1.predict(pr_predict.reshape(-1, 1))

plt.figure(figsize=(10,7))
plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(np.arange(0,len(data)),P, alpha = 0.75)
plt.plot(np.arange(0,len(data)),pr_predict, alpha = 0.75)
plt.plot(np.arange(0,len(data)),P1, alpha = 0.75)
plt.axvline(L,c='r')
#plt.xlim(18000,18500)
plt.show()

correction
r2_score(data['y'][L:].values,P[L:])
r2_score(data['y'][L:].values,pr_predict[L:])
r2_score(data['y'][L:].values,P1[L:])

# %% %%
PR = PeriodicRegression()
top_n = 25
PR.fit(data, top_n = top_n, max_correction = 300, cv=0.05, lags=[1,2,3,4,5,6,7], lag_freq='1D')
PR.plot_train_results()
PR.plot_train_results(x_lim = ('2020-06-01','2020-08-09'),
                      y_lim = (400,1800))
PR._scores

#PR.plot_spectrum(log=True)
	         MAE	MBE	              RMSE	      MAPE	    r2_score
train	98.179883	5.437123e-14	143.896124	0.117991	0.809861
test	117.154438	1.384111e+01	165.462281	0.140101	0.786918


MAE	MBE	RMSE	MAPE	r2_score
train	89.338889	-9.382801e-14	128.141860	10.987896	0.848517
test	96.992640	-1.005513e+01	131.613239	10.723795	0.868977
