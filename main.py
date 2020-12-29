import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

import functions
from PeriodicRegressionClass import PeriodicRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# %% %%
data = pd.read_csv('./data.csv')
# %% %%
#data = pd.read_csv('./data.csv')
data['dt'] = pd.to_datetime(data['timeFrom'])
data['y'] = data['energyKwh']
data = data[['dt','y']]

# %% %%
max_correction = 100
top_n = 3
result = pd.DataFrame([])
L = 3000
for x in range(1, max_correction, 1):
    PR = PeriodicRegression()
    PR.fit(data[:-L-x], top_n = top_n)
    MAX = PR.spectrum.iloc[PR.spectrum['abs'].idxmax()]
    r = {'x' : x,
         'freq' : np.real(MAX['freq']),
         'width' : np.real(MAX['width']),
         'abs' : np.real(MAX['abs'])}
    result = result.append(r,ignore_index=True)
peaks, _ = find_peaks(result['abs'])
optimal = np.arange(1, max_correction, 1)[peaks].min()
plt.plot(result['x'],result['abs'])
PR.plot_spectrum(log=True)
# %% %%
top_n = 3
correction = functions.find_length_correction(data[:-2999], max_correction=100, top_n=top_n, cv=0.1)
correction
# %% %%

data['y'] = data['y'].values#-restored
PR = PeriodicRegression()
PR.fit(data[:-3000-correction], top_n = top_n, cv=0.1)

PR.plot_spectrum(log = True)

restored = PR.predict(np.arange(0,len(data)))
restored = (restored-restored[:-3000-correction].min())/(restored[:-3000-correction].max()-restored[:-3000-correction].min())
restored = restored * (data['y'][:-3000-correction].max())#-data['y'][:-xxx].min())+data['y'][:-xxx].min()
plt.figure(figsize=(10,7))
plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(np.arange(0,len(data)),restored, alpha = 0.75)
#plt.xlim(18000,18500)
plt.show()



# %% %%
Spec = PR.spectrum.copy()
for i, row in Spec.iterrows():
    if (i==0)or(i==len(Spec)-1):
        pass
    else:
        A = row['abs']
        Am = Spec.loc[i-1,'abs']
        Ap = Spec.loc[i+1,'abs']
        if (A>Am)&(A>Ap):
            Spec.loc[i,'naive_peak'] = 1
        else:
            Spec.loc[i,'naive_peak'] = 0

def plot_spectrum(spectrum, top_n, save_to = None, log = False, scale = 1):
    top_spectrum = spectrum[spectrum['naive_peak']==1].sort_values(['abs']).tail(top_n)

    plt.figure(figsize=(10,7))
    plt.plot(spectrum['freq']*scale,
             spectrum['abs'], c='gray')
    plt.plot(spectrum[spectrum['naive_peak']==1]['freq']*scale,
             spectrum[spectrum['naive_peak']==1]['abs'],
             'x', c='b')
    plt.plot(top_spectrum['freq']*scale,
             top_spectrum['abs'],
             'x', c='r', markersize=10)
    if log==True:
        plt.xscale('log')
        plt.xlabel('Frequency (log scale), cph', fontsize=15)
    else:
        plt.xlabel('Frequency, cph', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Intencity, arb.units', fontsize=15)
    plt.title('Signal spectrum', fontsize=15)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()
Spec[Spec['naive_peak']==True].sort_values('int')

plot_spectrum(Spec, 3, log=True)

# %% %%
initial = pd.read_csv('/Users/wamiwamisoftware/Desktop/Work/slai/fourier-extrapolation-main/result.csv')
result = pd.read_csv('/Users/wamiwamisoftware/Desktop/Work/slai/fourier-extrapolation-main/result_tp.csv')

# %% %%
plt.figure(figsize=(10,7))
plt.plot(initial['energyKwh'])
plt.plot(result['periodicalFeature']+result['trend'], alpha=0.5)
plt.xlim(9000,10000)

