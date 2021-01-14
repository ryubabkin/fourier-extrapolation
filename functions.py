import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% %%
# Preparations
# %% %%

def data_init(df, date_format="%Y-%m-%d %H:%M:%S"):
    df['dt'] = pd.to_datetime(df['dt'], format = date_format)
    return df

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

def restore_TS(model, X_array):
    restored = restore_signal(spectrum = model.spectrum,
                              array = X_array,
                              top = model._top_n
                              ) / model._length
    restored += restore_trend(X_array, model._polyvals)
    return restored

def create_train_data(data, spectrum, top_n, lags, lag_freq):
    train_data = add_periodic_features(spectrum = spectrum,
                                       data = data,
                                       top = top_n)
    train_data = add_datetime_features(train_data)
    if lags is not None:
        train_data = add_lagged_features(data = train_data,
                                         lags = lags,
                                         freq = lag_freq).dropna()
    return train_data

# %% %%
# Missing data
# %% %%

def fill_missing(data):
    return data.fillna(0)


# %% %%
# Trend - Periodical decomposition
# %% %%

def get_trend(signal):
    polyvals = np.polyfit(x = np.arange(len(signal)),
                          y = np.array(signal),
                          deg = 1
                          )
    F = np.poly1d(polyvals.flatten())
    trend = F(np.arange(len(signal)))
    polyvals = polyvals.flatten().tolist()
    return trend, polyvals

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

def restore_trend(array, polyvals):
    restored = array*polyvals[0] + polyvals[1]
    return restored

def restore_signal(spectrum, array, top):
    spectrum = spectrum[spectrum['peak']==1].sort_values('abs', ascending=False).head(int(top))
    signal = np.zeros(len(array),dtype=np.complex_)
    for _, row in spectrum.iterrows():
        F = row['freq']
        P = row['phase']
        A = row['abs']
        signal += 2*A*np.cos(2*np.pi*F*array + P)
    return np.real(signal)

# %% %%
# Optimization
# %% %%

def find_length_correction(signal, max_correction, top_n, cv):
    result = pd.DataFrame([])
    for x in range(1, max_correction, 1):
        spectrum = get_frequencies(signal[:-x])
        MAX = spectrum.iloc[spectrum['abs'].idxmax()]
        r = {'x' : x,
             'freq' : np.real(MAX['freq']),
             'width' : np.real(MAX['width']),
             'abs' : np.real(MAX['abs'])}
        result = result.append(r,ignore_index=True)
    optimal = int(result[result['abs']==result['abs'].max()]['x'].min())
    return optimal, result

def define_optimal_n(signal, cv = 0.1, n_max = 20):
    if (cv >= 1) or (cv <= 0):
        cv = 0.1
    len_train = int(len(signal)*(1-cv))
    X = np.arange(0,len(signal))
    spectrum = get_frequencies(signal[:len_train])
    RESULT = pd.DataFrame()
    for n in range(1,n_max):
        restored = restore_signal(spectrum = spectrum,
                                  array = X,
                                  top = n)/len(X)
        RESULT = RESULT.append({
            "n" : n,
            "mae_train" : MAE(signal[:len_train],restored[:len_train]),
            "mae_cv" : MAE(signal[len_train:],restored[len_train:])
        }, ignore_index = True)
    optimal_n = RESULT.iloc[RESULT[RESULT['n']>2]['mae_cv'].idxmin()]['n']
    return int(optimal_n), RESULT



# %% %%
# Training functions
# %% %%

def train_regression(data, cv):
    train = data.iloc[:-int(len(data)*cv)].reset_index(drop=True)
    test = data.iloc[-int(len(data)*cv):].reset_index(drop=True)
    regressor = LinearRegression().fit(train.drop(['dt','y'], axis = 1),
                                       train['y'])
    train['y_pred'] = regressor.predict(train.drop(['dt','y'], axis = 1))
    test['y_pred'] = regressor.predict(test.drop(['dt','y'], axis = 1))
    scores = result_scores(train['y'].values, test['y'].values,
                           train['y_pred'].values, test['y_pred'].values )
    result = {}
    result['train'] = train[['dt','y','y_pred']]
    result['test'] = test[['dt','y','y_pred']]
    return regressor, scores, result


# %% %%
# Plottings
# %% %%

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
        plt.xlabel('Frequency (log scale) [1 / Day]', fontsize=15)
    else:
        plt.xlabel('Frequency [1 / Day]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Intencity [arb.units]', fontsize=15)
    plt.title('Signal spectrum', fontsize=15)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()

def plot_train_results(results, x_lim = None, y_lim = None, save_to = None):
    train = results['train']
    test = results['test']

    plt.figure(figsize=(12,5))
    plt.plot(train['dt'],
             train['y'], c='gray')
    plt.plot(test['dt'],
             test['y'], c='gray')
    plt.plot(train['dt'],
             train['y_pred'], c='blue')
    plt.plot(test['dt'],
             test['y_pred'], c='orange')
    plt.xlabel('Timestamp', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Target value', fontsize=15)
    plt.title('Training results', fontsize=15)
    if x_lim:
        plt.xlim(pd.to_datetime(x_lim[0]),pd.to_datetime(x_lim[1]))
    if y_lim:
        plt.ylim(y_lim[0],y_lim[1])
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
    plt.show()

def plot_corrections(corrections, save_to = None):
    plt.figure(figsize=(10,5))
    optimal = int(corrections[corrections['abs']==corrections['abs'].max()]['x'].min())
    plt.plot(corrections[corrections['x']==optimal]['x'],
             corrections[corrections['x']==optimal]['abs'],
             'x', c='r', markersize=15, markeredgewidth=5)
    plt.plot(corrections['x'],
             corrections['abs'], c='b')
    plt.xlabel('Shifts', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Main frequency amplitude', fontsize=15)
    plt.title('The optimal cut length is '+str(optimal), fontsize=15)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
    plt.show()

# %% %%
# Features
# %% %%

def add_periodic_features(spectrum, data, top):
    array =  np.arange(len(data))
    spectrum = spectrum[spectrum['peak']==1].sort_values('abs', ascending=False).head(int(top))
    i = 1
    for freq in spectrum['freq'].unique():
        data['freq'+str(i)+'_sin'] = np.sin(2*np.pi*freq*array)
        data['freq'+str(i)+'_cos'] = np.cos(2*np.pi*freq*array)
        i += 1
    return data

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
"""
def add_lagged_features(df, lags, forward = 0, pred = 0):
    lags = list(map(int, lags))
    df_lags = pd.DataFrame([])
    for column in df.columns:
        for lag in lags:
            if (pred == 0)&(df[column].nunique() > 1) or (pred == 1):
                df_lags[column + '_lag_' + str(lag)] = df[column].rolling(lag).mean().shift(forward).fillna(0)
            else:
                pass
    return(df_lags)
"""
def add_lagged_features(data, lags, freq = None):
    if freq == None:
        for lag in lags:
            data[f'lag_{lag}'] = data['y'].shift(lag).values
    else:
        data = data.set_index('dt')
        for lag in lags:
            shift = data['y'].shift(lag,freq).rename(f'lag_{lag}_{freq}')
            data = data.join(shift)
        data = data.reset_index()
    return data

def features_imp(df, target):
    from sklearn.ensemble import RandomForestRegressor as RF
    df['RAND_bin'] = np.random.randint(2, size = len(df[target]))
    df['RAND_uniform'] = np.random.uniform(0,1, len(df[target]))
    df['RAND_int'] = np.random.randint(100, size = len(df[target]))
    columns = df.drop(target, axis = 1).columns.tolist()
    estimator = RF(random_state = 42, n_estimators = 50)
    estimator.fit(df[columns], df[target])
    Y_pred = estimator.predict(df[columns])
    baseline = MAE(Y_pred,df[target])
    imp = []
    for col in columns:
        col_imp = []
        for n in range(3):
            save = df[col].copy()
            df[col] = np.random.permutation(df[col])
            Y_pred = estimator.predict(df[columns])
            m = MAE(Y_pred, df[target])
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

# %% %%
# Metrics
# %% %%

def MAE(A,F):
    return np.mean(np.fabs(A-F))

def MBE(A,F):
    return np.mean(A-F)

def MAPE(A,F):
    mask = A != 0
    return np.mean((np.fabs(A - F)/A)[mask])*100

def RMSE(A,F):
    return np.sqrt(np.mean((A-F)**2))

def result_scores(y_train, y_test, y_train_pred, y_test_pred):
    scores = pd.DataFrame([], index = ['train','test'])
    scores.loc['train','MAE'] = MAE(y_train, y_train_pred)
    scores.loc['train','MBE'] = MBE(y_train, y_train_pred)
    scores.loc['train','RMSE'] = RMSE(y_train, y_train_pred)
    scores.loc['train','MAPE'] = MAPE(y_train, y_train_pred)
    scores.loc['train','r2_score'] = r2_score(y_train, y_train_pred)
    scores.loc['test','MAE'] = MAE(y_test, y_test_pred)
    scores.loc['test','MBE'] = MBE(y_test, y_test_pred)
    scores.loc['test','RMSE'] = RMSE(y_test, y_test_pred)
    scores.loc['test','MAPE'] = MAPE(y_test, y_test_pred)
    scores.loc['test','r2_score'] = r2_score(y_test, y_test_pred)
    return scores
