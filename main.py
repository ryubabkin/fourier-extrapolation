import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import functions
# %% %%
data_all = pd.read_csv('/home/brom/LAB/SLai_Folder/SLai.model/energy_records_history.csv')
for id in data_all['id'].unique():
    df = data_all[data_all['id']==id]
    df['timeFrom'] = pd.to_datetime(df['timeFrom'])
    print(id)
    plt.plot(df['timeFrom'],df['energyKwh'])
    plt.show()

# %% %%
id = 150151
data = data_all[data_all['id']==id]
#data = pd.read_csv('./data.csv')
data['dt'] = pd.to_datetime(data['timeFrom'])
data['y'] = data['energyKwh']
data = data[['dt','y']]

# %% %%
def data_init(df, date_format="%Y-%m-%d %H:%M:%S"):
    df['dt'] = pd.to_datetime(df['dt'], format = date_format)
    return df

def get_trend(signal, deg = 1):
    polyvals = np.polyfit(x = np.arange(len(signal)),
                          y = np.array(signal),
                          deg = deg
                          )
    F = np.poly1d(polyvals.flatten())
    trend = F(np.arange(len(signal)))
    polyvals = polyvals.flatten().tolist()
    return trend, polyvals

class SLRegression(object):

    def __init__(self,
                 time_step = '1h',
                 n_freqs = 10):
        return

    def fit(self, data,
            date_format = "%Y-%m-%d %H:%M:%S"):

        data = data_init(data)
        ts, self._dt_freq = functions.get_time_series(data)
        self._dt_start = ts['dt'].min()
        self._dt_end = ts['dt'].max()
        ts.fillna(0, inplace=True)
        ts = functions.add_datetime_features(ts)
        self._trend, self._polyvals = get_trend(ts['y'])
        self.data = ts


SLR = SLRegression()
SLR.fit(data)
df_ts = SLR.data.copy()
trend = SLR.trend

# %% %%
def MAE(E,F):
    mae = np.mean(abs(E-F))
    return mae

# %% %%

signal = df_ts['y'].values - trend
top_n = 'auto'
n_max = 20
cv = 0.1

def get_frequencies(signal):
    signal = np.round(signal,2)
    spectrum = pd.DataFrame([])
    spectrum['freq'] = np.fft.rfftfreq(signal.size)
    spectrum['int'] = np.fft.rfft(spectrum)
    spectrum['abs'] = np.abs(spectrum['int'])
    spectrum['phase'] = np.angle(spectrum['int'])
    peaks, _ = find_peaks(spectrum['abs'], height=0)
    spectrum['peak'] = np.where(spectrum.index.isin(peaks),1,0)
    return spectrum

def restore_signal(spectrum, array, top):
    spectrum = spectrum[spectrum['peak']==1].sort_values('abs').tail(top)
    signal = np.zeros(len(array))
    for _, row in spectrum.iterrows():
        F = abs(row['freq'])
        P = abs(row['phase'])
        A = abs(row['abs'])
        signal += 2*A*np.cos(2*np.pi*F*array + P)
    return signal

def define_optimal_n(signal, cv = 0.1, n_max = 20):
    if (cv >= 1) or (cv <= 0):
        cv = 0.1
    len_train = int(len(signal)*(1-cv))
    X = np.arange(0,len(signal))
    spectrum = get_frequencies(signal[:len_train])
    RESULT = pd.DataFrame()
    for n in range(1,n_max):
        restored = restore_signal(spectrum, X, n)/len(X)
        RESULT = RESULT.append({
            "n" : n,
            "mae_train" : MAE(signal[:len_train],restored[:len_train]),
            "mae_cv" : MAE(signal[len_train:],restored[len_train:])
        }, ignore_index = True)
    optimal_n = RESULT.iloc[RESULT[RESULT.index>0]['mae_cv'].idxmin()]['n']
    return optimal_n, RESULT

if top_n == 'auto':
    top_n, mae_result = define_optimal_top(signal, cv = cv, n_max = n_max)


spectrum = get_frequencies(signal)
restored

#plt.figure(figsize=(15,5))
#plt.plot(X_ext, signal)
#plt.plot(X_ext, S/len(X))

# %% %%
