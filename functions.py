import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# %% %%
def get_time_series(df, freq='auto'):
    if freq == 'auto':
        freq = int(df['dt'].diff().dt.total_seconds().mode().values[0])
        freq = str(freq)+'s'
    df_ts = pd.DataFrame([])
    df_ts['dt'] = pd.date_range(
                    start = df['dt'].min(),
                    end = df['dt'].max(),
                    freq = freq )
    return df_ts.merge(df, how='left', on = 'dt'), freq

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
        signal += 2*A*np.exp(1j*2*np.pi*F*array + 1j*P)
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


def MAPE(E,F,C):
    sigma = np.sum(abs(F - E)/C/4)/len(F)
    return sigma
def MAE(E,F):
    sigma = np.sum(abs(F - E))/len(F)
    return sigma
def MABE(E,F):
    sigma = np.sum(F - E)/len(F)
    return sigma


def plot_spectrum(spectrum, top_n, save_to = None, log = False, scale = 1):
    top_spectrum = spectrum[spectrum['peak']==1].sort_values(['abs']).tail(top_n)

    plt.figure(figsize=(10,7))
    plt.plot(spectrum['freq']*scale,
             spectrum['abs'], c='gray')
    plt.plot(spectrum[spectrum['peak']==1]['freq']*scale,
             spectrum[spectrum['peak']==1]['abs'],
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

def add_datetime_features(data):
    dt = data['dt'].dt
    ## hours.mins.seconds into circle coordinates hour_x and hour_y
    data['time_s'] = np.sin(2.*np.pi*(dt.hour+dt.minute/60.0+dt.second/3600.0)/24.)
    data['time_c'] = np.cos(2.*np.pi*(dt.hour+dt.minute/60.0+dt.second/3600.0)/24.)
    ## days of week into circle coordinates day_month_x and day_month_y
    data['day_month_s'] = np.sin(2.*np.pi*dt.day/dt.daysinmonth)
    data['day_month_c'] = np.cos(2.*np.pi*dt.day/dt.daysinmonth)
    ## days of week into circle coordinates day_week_x and day_week_y
    data['day_week_s'] = np.sin(2.*np.pi*dt.dayofweek/7.)
    data['day_week_c'] = np.cos(2.*np.pi*dt.dayofweek/7.)
    ## month into circle coordinates month_x and month_y
    data['month_s'] = np.sin(2.*np.pi*dt.month/12.)
    data['month_c'] = np.cos(2.*np.pi*dt.month/12.)
    return data

def lagged_features(df, lags, forward = 0, pred = 0):
    lags = list(map(int, lags))
    df_lags = pd.DataFrame([])
    for column in df.columns:
        for lag in lags:
            if (pred == 0)&(df[column].nunique() > 1) or (pred == 1):
                df_lags[column + '_lag_' + str(lag)] = df[column].rolling(lag).mean().shift(forward).fillna(0)
            else:
                pass
    return(df_lags)

def features_imp(df, target):
    from sklearn.ensemble import RandomForestRegressor as RF
    from sklearn.metrics import mean_absolute_error as metric
    df['RAND_bin'] = np.random.randint(2, size = len(df[target]))
    df['RAND_uniform'] = np.random.uniform(0,1, len(df[target]))
    df['RAND_int'] = np.random.randint(100, size = len(df[target]))
    columns = df.drop(target, axis = 1).columns.tolist()
    estimator = RF(random_state = 42, n_estimators = 50)
    estimator.fit(df[columns], df[target])
    Y_pred = estimator.predict(df[columns])
    baseline = metric(Y_pred,df[target])
    imp = []
    for col in columns:
        col_imp = []
        for n in range(3):
            save = df[col].copy()
            df[col] = np.random.permutation(df[col])
            Y_pred = estimator.predict(df[columns])
            m = metric(Y_pred, df[target])
            df[col] = save
            col_imp.append(baseline - m)
        imp.append(np.mean(col_imp))
    FI = pd.DataFrame([])
    FI['feature'] = columns
    FI['value'] = np.array(imp)
    FI['value'] = -FI['value']# / FI['value'].sum()
    FI = FI.sort_values('value', ascending = False).reset_index(drop = True)
    threshold = FI[FI['feature'].isin(['RAND_bin','RAND_int','RAND_uniform'])]['value'].max()
    FI['important'] = np.where(FI['value']>threshold, True, False)
    return(FI)
