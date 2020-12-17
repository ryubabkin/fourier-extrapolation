import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

import functions
from PeriodicRegressionClass import PeriodicRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# %% %%
data_all = pd.read_csv('./energy_records_history.csv')
#for id in data_all['id'].unique():
#    df = data_all[data_all['id']==id]
#    df['timeFrom'] = pd.to_datetime(df['timeFrom'])
#    print(id)
#    plt.plot(df['timeFrom'],df['energyKwh'])
#    plt.show()

# %% %%
id = 150150
data = data_all[data_all['id']==id]
#data = pd.read_csv('./data.csv')
data['dt'] = pd.to_datetime(data['timeFrom'])
data['y'] = data['energyKwh']
data = data[['dt','y']]

# %% %%

# %% %%
#WWW = []
#for xxx in np.arange(10,100,1):
xxx = 110
#data = pd.DataFrame([])
#X = np.arange(0,xxx,0.1)
#data['dt'] = pd.date_range(start = '2020-01-01', periods = len(X), freq = '1s')
#f1 = 0.1
#f2 = 0.5
#signal = np.cos(2*np.pi*f1*X)*10 + np.sin(2*np.pi*f2*X)
#data['y'] = signal
WWW = pd.DataFrame([])
for xxx in range(0,1000):
    PR = PeriodicRegression()
    PR.fit(data[xxx:], top_n = 3)
    #PR.plot_spectrum(log=False)
    #restored = PR.predict(np.arange(xxx,len(data)))

    #plt.plot(np.arange(0,len(data)),data['y'])
    #plt.plot(np.arange(xxx,len(data)),restored, alpha = 0.75)
    #plt.show()
    #
    SPEC = PR.spectrum[PR.spectrum['peak']==1].sort_values('abs').tail(3)

    W = {'x' : xxx,
     'f1' : SPEC['freq'].values[0],
     'w1' : SPEC['width'].values[0],
     'f2' : SPEC['freq'].values[1],
     'w2' : SPEC['width'].values[1],
     'f3' : SPEC['freq'].values[2],
     'w3' : SPEC['width'].values[2]}
    WWW = WWW.append(W,ignore_index=True)

WWW
# %% %%
plt.figure(figsize=(10,5))

plt.plot(WWW[['w1','w2','w3']].mean(axis=1))
WWW['w2'].idxmin()
WWW[['w1','w2','w3']].mean(axis=1).idxmin()
# %% %%
xxx=0
data['y'] = data['y'].values#-restored
PR = PeriodicRegression()
PR.fit(data[xxx:], top_n = 10, cv=0.1)
#PR.plot_spectrum(log = False)
restored = PR.predict(np.arange(xxx,len(data)))

plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(np.arange(0,len(data)),restored, alpha = 0.75)
plt.xlim(20000,30000)
plt.show()

functions.MAE(data['y'].values[xxx:],restored)
plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(data['y'].values-restored)
#plt.xlim(10000,12000)
