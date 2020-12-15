import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

import functions
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# %% %%
data_all = pd.read_csv('./energy_records_history.csv')
for id in data_all['id'].unique():
    df = data_all[data_all['id']==id]
    df['timeFrom'] = pd.to_datetime(df['timeFrom'])
    print(id)
    plt.plot(df['timeFrom'],df['energyKwh'])
    plt.show()

# %% %%
id = 150150
data = data_all[data_all['id']==id]
#data = pd.read_csv('./data.csv')
data['dt'] = pd.to_datetime(data['timeFrom'])
data['y'] = data['energyKwh']
data = data[['dt','y']]

# %% %%
def MAE(E,F):
    sigma = np.sum(abs(F - E))/len(F)
    return sigma
def data_init(df, date_format="%Y-%m-%d %H:%M:%S"):
    df['dt'] = pd.to_datetime(df['dt'], format = date_format)
    return df

def get_trend(signal):
    polyvals = np.polyfit(x = np.arange(len(signal)),
                          y = np.array(signal),
                          deg = 1
                          )
    F = np.poly1d(polyvals.flatten())
    trend = F(np.arange(len(signal)))
    polyvals = polyvals.flatten().tolist()
    return trend, polyvals

def restore_trend(array, polyvals):
    restored = array*polyvals[0] + polyvals[1]
    return restored

def get_frequencies(signal):
    signal = np.round(signal,2)
    spectrum = pd.DataFrame([])
    spectrum['freq'] = np.fft.rfftfreq(signal.size)
    spectrum['int'] = np.fft.rfft(signal)
    spectrum['abs'] = np.abs(spectrum['int'])
    spectrum['phase'] = np.angle(spectrum['int'])
    peaks, _ = find_peaks(spectrum['abs'], height=0)
    half_width = peak_widths(spectrum['abs'], peaks, rel_height=0.5)[0]
    spectrum['peak'] = np.where(spectrum.index.isin(peaks),1,0)
    spectrum.loc[spectrum.index[peaks], 'width'] = half_width
    return spectrum

def restore_signal(spectrum, array, top):
    spectrum = spectrum[spectrum['peak']==1].sort_values('abs').tail(int(top))
    signal = np.zeros(len(array),dtype=np.complex_)
    for _, row in spectrum.iterrows():
        F = row['freq']
        P = row['phase']
        A = row['abs']
        #signal += 2*A*np.cos(2*np.pi*F*array + P)
        signal += 2*A*np.exp(1j*2*np.pi*F*array + 1j*P)
    #zero = abs(spectrum.iloc[0])
    #signal += 2*zero['abs']*np.cos(2*np.pi*zero['freq']*array - zero['phase'])
    return np.real(signal)

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
    optimal_n = RESULT.iloc[RESULT[RESULT['n']>2]['mae_cv'].idxmin()]['n']
    return int(optimal_n), RESULT

# %% %%
class PeriodicRegression(object):

    def __init__(self,
                 time_step = '1h',
                 n_freqs = 10):
        return

    def fit(self, data,
            top_n = 'auto',
            n_max = 20,
            cv = 0.1,
            date_format = "%Y-%m-%d %H:%M:%S"):
        data = data_init(df = data,
                         date_format = date_format
                         )
        self.time_series, self._dt_freq = functions.get_time_series(data)
        self._dt_start = self.time_series['dt'].min()
        self._dt_end = self.time_series['dt'].max()

        self.time_series.fillna(0, inplace=True)

        self.time_series = functions.add_datetime_features(self.time_series)
        self._trend, self._polyvals = get_trend(self.time_series['y'])
        signal = self.time_series['y'] - self._trend

        if top_n == 'auto':
            self._top_n, self._mae_score = define_optimal_n(signal = signal,
                                                            cv = cv,
                                                            n_max = n_max
                                                            )
        else:
            self._top_n = int(top_n)
            self._mae_score = None

        self.spectrum = get_frequencies(signal = signal)
        self._length = len(signal)

    def predict(self, X_array):
        restored = restore_signal(spectrum = self.spectrum,
                                  array = X_array,
                                  top = self._top_n
                                  ) / self._length
        restored += restore_trend(X_array, self._polyvals)
        return restored

    def plot_spectrum(self, save_to = None, log = False):
        scale = self._length / (self._dt_end - self._dt_start).total_seconds() * 3600
        top_spectrum = self.spectrum[self.spectrum['peak']==1].sort_values(['abs']).tail(self._top_n)

        plt.figure(figsize=(10,7))
        plt.plot(self.spectrum['freq']*scale,
                 self.spectrum['abs'], c='gray')
        plt.plot(self.spectrum[self.spectrum['peak']==1]['freq']*scale,
                 self.spectrum[self.spectrum['peak']==1]['abs'],
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
PR = PeriodicRegression()
PR.fit(data[xxx:], top_n = "auto", cv=0.1)
PR.plot_spectrum(log=False)
PR._mae_score
restored = PR.predict(np.arange(xxx,len(data)+1000))

plt.plot(np.arange(0,len(data)),data['y'])
plt.plot(np.arange(xxx,len(data)+1000),restored, alpha = 0.75)
plt.xlim(40000,50000)
plt.show()

MAE(data['y'].values[xxx:],restored[:-1000])

PR._mae_score.plot()
