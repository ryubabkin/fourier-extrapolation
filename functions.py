import warnings

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, to_datetime, date_range, Timedelta, concat
from scipy.signal import find_peaks, peak_widths
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Preparations
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def data_init(df, date_format="%Y-%m-%d %H:%M:%S"):
    df['dt'] = to_datetime(df['dt'], format=date_format)
    return df


def get_time_series(df, freq='auto'):
    df = df.groupby('dt').mean().reset_index()
    if freq == 'auto':
        freq = int(df['dt'].diff().dt.total_seconds().mode().values[0])
        freq = str(freq) + 's'
    df_ts = DataFrame([])
    df_ts['dt'] = date_range(start=df['dt'].min(),
                             end=df['dt'].max(),
                             freq=freq)
    return df_ts.merge(df, how='left', on='dt'), freq


"""
def restore_TS(model, x_array):
    restored = restore_signal(spectrum=model.spectrum,
                              array=x_array,
                              top=model._top_n
                              ) / model._length
    restored += restore_trend(x_array, model._polyvals)
    return restored
"""


def create_train_data(data, spectrum, top_n, lags, lag_freq):
    train_data = add_periodic_features(spectrum=spectrum,
                                       data=data,
                                       top=top_n)
    train_data = add_datetime_features(train_data)
    if lags is not None:
        train_data = add_lagged_features(data=train_data,
                                         lags=lags,
                                         freq=lag_freq).dropna()
    return train_data


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Missing data
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def get_deltas(data, freq):
    data = data.groupby('dt').mean().reset_index().dropna(subset=['y'])
    deltas = DataFrame([])
    deltas['dt'] = data['dt']
    deltas['diff'] = data['dt'] - data['dt'].shift()
    deltas = deltas[deltas['diff'] != freq][1:]
    deltas.index = (deltas.index - deltas['diff'] / freq).astype(int)
    deltas['dt'] = deltas['dt'] - deltas['diff'] + freq
    deltas['steps'] = (deltas['diff'] / freq - 1).astype(int)
    deltas['diff'] = deltas['diff'] - freq
    return deltas


def get_interpolation(data, start, steps, freq):
    result = DataFrame([])
    for i in range(0, steps):
        date = start + freq * i
        missed = DataFrame([])
        dtrg = date_range(date - Timedelta('1d') - Timedelta('1d') * (freq * i).days,
                          date + Timedelta('1d') + Timedelta('1d') * (freq * (steps - i)).days,
                          freq='1d')
        missed['dt'] = dtrg
        for point in dtrg:
            missed.loc[missed['dt'] == point, 'y'] = data.loc[data['dt'] == point, 'y'].values[0]
        missed['y'] = missed['y'].interpolate(method='linear')
        result = concat([result, missed])
    result = result.sort_values('dt')
    result = result.drop_duplicates('dt')
    result.reset_index(drop=True, inplace=True)
    return result


def fill_missing(data, freq, ranges=None):
    freq = Timedelta(freq)
    if ranges is None:
        ranges = ['3h', '3d']
    short_range = Timedelta(ranges[0])
    long_range = Timedelta(ranges[1])
    deltas = get_deltas(data, freq)
    for _, row in deltas[deltas['diff'] <= short_range].iterrows():
        start = row['dt'] - freq
        end = row['dt'] + (row['steps'] + 1) * freq
        data.loc[(data['dt'] >= start) & (data['dt'] < end), 'y'] = data.loc[
            (data['dt'] >= start) & (data['dt'] < end), 'y'].interpolate(method='linear')
    for _, row in deltas[(deltas['diff'] > short_range) & (deltas['diff'] <= long_range)].iterrows():
        start = row['dt']
        steps = row['steps']
        interp = get_interpolation(data, start, steps, freq)
        for time in interp['dt']:
            data.loc[data['dt'] == time, 'y'] = interp.loc[interp['dt'] == time, 'y'].values[0]
    return data, deltas


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Trend - Periodical decomposition
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def get_trend(signal):
    polyvals = np.polyfit(x=np.arange(len(signal)),
                          y=np.array(signal),
                          deg=1
                          )
    flat = np.poly1d(polyvals.flatten())
    trend = flat(np.arange(len(signal)))
    polyvals = polyvals.flatten().tolist()
    return trend, polyvals


def get_frequencies(signal):
    signal = np.round(signal, 2)
    spectrum = DataFrame([])
    spectrum['freq'] = np.fft.rfftfreq(signal.size)
    spectrum['int'] = np.fft.rfft(signal)
    spectrum['abs'] = np.abs(spectrum['int'])
    spectrum['phase'] = np.angle(spectrum['int'])
    peaks, _ = find_peaks(spectrum['abs'], height=0)
    half_width = peak_widths(spectrum['abs'], peaks, rel_height=0.5)[0]
    spectrum['peak'] = np.where(spectrum.index.isin(peaks), 1, 0)
    spectrum.loc[spectrum.index[peaks], 'width'] = half_width
    return spectrum


def restore_trend(array, polyvals):
    restored = array * polyvals[0] + polyvals[1]
    return restored


def restore_signal(spectrum, array, top):
    spectrum = spectrum[spectrum['peak'] == 1]
    spectrum = spectrum.sort_values('abs', ascending=False).head(int(top))
    signal = np.zeros(len(array), dtype=np.complex_)
    for _, row in spectrum.iterrows():
        freq = row['freq']
        phase = row['phase']
        intensity = row['abs']
        signal += 2 * intensity * np.cos(2 * np.pi * freq * array + phase)
    return np.real(signal)


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Optimization
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def find_length_correction(signal, max_correction, cv):
    result = DataFrame([])
    for x in range(1, max_correction, 1):
        spectrum = get_frequencies(signal[x:-cv-1])
        max_unit = spectrum.iloc[spectrum['abs'].idxmax()]
        r = {'x': x,
             'freq': np.real(max_unit['freq']),
             'width': np.real(max_unit['width']),
             'abs': np.real(max_unit['abs'])}
        result = result.append(r, ignore_index=True)
    optimal = int(result[result['abs'] == result['abs'].max()]['x'].min())
    return optimal


def define_optimal_n(signal, cv, n_max=20):
    len_train = len(signal) - cv
    array = np.arange(0, len(signal))
    spectrum = get_frequencies(signal[:len_train])
    result = DataFrame()
    for n in range(1, n_max):
        restored = restore_signal(spectrum=spectrum,
                                  array=array,
                                  top=n) / len(array)
        result = result.append({
            "n": n,
            "mae_train": MAE(signal[:len_train], restored[:len_train]),
            "mae_cv": MAE(signal[len_train:], restored[len_train:])
        }, ignore_index=True)
    optimal_n = result.iloc[result[result['n'] > 2]['mae_cv'].idxmin()]['n']
    return int(optimal_n), result

def correct_top_spectrum(signal, top_n, max_correction, cv):
    corrected_top_spectrum = DataFrame()
    for n in range(top_n):
        correction_cut  = find_length_correction(signal=signal,
                                                     max_correction=max_correction,
                                                     cv=cv)

        spectrum = get_frequencies(signal=signal[correction_cut:-cv-1])
        corrected_top_spectrum = concat([corrected_top_spectrum, spectrum[spectrum['peak']==1].sort_values('abs').tail(1)])
        df = add_periodic_features(spectrum, signal.to_frame(), 1)
        regressor = LinearRegression().fit(df.drop(['y'], axis=1)[:-cv-1],
                                           df['y'][:-cv-1])
        line = regressor.predict(df.drop(['y'], axis=1))
        signal = signal - line
    return corrected_top_spectrum


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Features
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def add_periodic_features(spectrum, data, top):
    array = np.arange(len(data))
    spectrum = spectrum[spectrum['peak'] == 1]
    spectrum = spectrum.sort_values('abs', ascending=False).head(int(top))
    i = 1

    for freq in spectrum['freq'].unique():
        data['freq' + str(i) + '_sin'] = np.sin(2 * np.pi * freq * array)
        data['freq' + str(i) + '_cos'] = np.cos(2 * np.pi * freq * array)
        i += 1
    return data

def add_datetime_features(data):
    dt = data['dt'].dt
    # hours.minutes.seconds into circle coordinates hour_x and hour_y
    data['time_sin'] = np.sin(2. * np.pi * (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.)
    data['time_cos'] = np.cos(2. * np.pi * (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.)
    # days of week into circle coordinates day_month_x and day_month_y
    data['day_of_month_sin'] = np.sin(2. * np.pi * dt.day / dt.daysinmonth)
    data['day_of_month_cos'] = np.cos(2. * np.pi * dt.day / dt.daysinmonth)
    # days of week into circle coordinates day_week_x and day_week_y
    data['day_of_week_sin'] = np.sin(2. * np.pi * dt.dayofweek / 7.)
    data['day_of_week_cos'] = np.cos(2. * np.pi * dt.dayofweek / 7.)
    # month into circle coordinates month_x and month_y
    data['month_sin'] = np.sin(2. * np.pi * dt.month / 12.)
    data['month_cos'] = np.cos(2. * np.pi * dt.month / 12.)
    return data


def add_lagged_features(data, lags, freq=None):
    if freq is None:
        for lag in lags:
            data[f'lag_{lag}'] = data['y'].shift(lag).values
    else:
        data = data.set_index('dt')
        for lag in lags:
            shift = data['y'].shift(lag, freq).rename(f'lag_{lag}_{freq}')
            data = data.join(shift)
        data = data.reset_index()
    return data


def features_imp(df, target):
    from sklearn.ensemble import RandomForestRegressor as RF
    df['RAND_bin'] = np.random.randint(2, size=len(df[target]))
    df['RAND_uniform'] = np.random.uniform(0, 1, len(df[target]))
    df['RAND_int'] = np.random.randint(100, size=len(df[target]))
    columns = df.drop(target, axis=1).columns.tolist()
    estimator = RF(n_estimators=50)
    estimator.fit(df[columns], df[target])
    y_pred = estimator.predict(df[columns])
    baseline = MAE(y_pred, df[target])
    imp = []
    for col in columns:
        col_imp = []
        for n in range(3):
            save = df[col].copy()
            df[col] = np.random.permutation(df[col])
            y_pred = estimator.predict(df[columns])
            m = MAE(y_pred, df[target])
            df[col] = save
            col_imp.append(baseline - m)
        imp.append(np.mean(col_imp))
    FI = DataFrame([])
    FI['feature'] = columns
    FI['value'] = -np.array(imp)
    FI = FI.sort_values('value', ascending=False).reset_index(drop=True)
    M = FI[FI['feature'].isin(['RAND_bin', 'RAND_int', 'RAND_uniform'])]['value'].max()
    S = FI[FI['feature'].isin(['RAND_bin', 'RAND_int', 'RAND_uniform'])]['value'].std()
    threshold = M + S
    FI['important'] = np.where(FI['value'] > threshold, True, False)
    return (FI)


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Training functions
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def train_regression(data, cv):
    train = data.iloc[:-cv-1].reset_index(drop=True)
    test = data.iloc[-cv-1:].reset_index(drop=True)
    regressor = LinearRegression().fit(train.drop(['dt', 'y'], axis=1),
                                       train['y'])
    train['y_pred'] = regressor.predict(train.drop(['dt', 'y'], axis=1))
    test['y_pred'] = regressor.predict(test.drop(['dt', 'y'], axis=1))
    scores = result_scores(train['y'].values, test['y'].values,
                           train['y_pred'].values, test['y_pred'].values)
    result = {'train': train[['dt', 'y', 'y_pred']],
              'test': test[['dt', 'y', 'y_pred']]
              }
    return regressor, scores, result


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Plot functions
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def plot_spectrum(spectrum, top_n, save_to=None, log=False, scale=1):
    top_spectrum = spectrum[spectrum['peak'] == 1].sort_values(['abs']).tail(top_n)

    plt.figure(figsize=(10, 7))
    plt.plot(spectrum['freq'] * scale,
             spectrum['abs'], c='gray')
    plt.plot(spectrum[spectrum['peak'] == 1]['freq'] * scale,
             spectrum[spectrum['peak'] == 1]['abs'],
             'x', c='b')
    plt.plot(top_spectrum['freq'] * scale,
             top_spectrum['abs'],
             'x', c='r', markersize=10)
    if log:
        plt.xscale('log')
        plt.xlabel('Frequency (log scale) [1 / Day]', fontsize=15)
    else:
        plt.xlabel('Frequency [1 / Day]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Intensity [arb.units]', fontsize=15)
    plt.title('Signal spectrum', fontsize=15)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_train_results(results, x_lim=None, y_lim=None, save_to=None):
    train = results['train']
    test = results['test']

    plt.figure(figsize=(12, 5))
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
        plt.xlim(to_datetime(x_lim[0]), to_datetime(x_lim[1]))
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_corrections(corrections, save_to=None):
    plt.figure(figsize=(10, 5))
    optimal = int(corrections[corrections['abs'] == corrections['abs'].max()]['x'].min())
    plt.plot(corrections[corrections['x'] == optimal]['x'],
             corrections[corrections['x'] == optimal]['abs'],
             'x', c='r', markersize=15, markeredgewidth=5)
    plt.plot(corrections['x'],
             corrections['abs'], c='b')
    plt.xlabel('Shifts', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Main frequency amplitude', fontsize=15)
    plt.title('The optimal cut length is ' + str(optimal), fontsize=15)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_missing_data(data, missing, freq, frame=None, save_to=None):
    freq = Timedelta(freq)
    if frame is None:
        frame = 5
    for _, row in missing.iterrows():
        start = row['dt']
        end = row['dt'] + (row['steps'] + 1) * freq
        missed_data = data[(data['dt'] >= start) & (data['dt'] < end)]

        start_frame = start - freq * frame
        end_frame = end + freq * frame
        whole_data = data[(data['dt'] >= start_frame) & (data['dt'] < end_frame)]

        plt.figure(figsize=(12, 5))
        plt.plot(whole_data['dt'], whole_data['y'], c='b', label='original data')
        plt.plot(missed_data['dt'], missed_data['y'], 'o-', c='r', label='filled data')
        plt.legend(fontsize=15, loc=1)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=13)
        plt.title(f"{start.strftime(format='%Y-%m-%d %H:%M:%S')} / {row['steps']} steps",
                  fontsize=15)
        plt.tight_layout()
        name = '/missed_' + start.strftime(format='%Y-%m-%d %H:%M:%S') + '.png'
        if save_to:
            plt.savefig(save_to + name)
        plt.show()


# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%
# Metrics
# %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %% %%

def MAE(A, F):
    return np.mean(np.fabs(A - F))


def MBE(A, F):
    return np.mean(A - F)


def MAPE(A, F):
    mask = A != 0
    return np.mean((np.fabs(A - F) / A)[mask]) * 100


def RMSE(A, F):
    return np.sqrt(np.mean((A - F) ** 2))


def result_scores(y_train, y_test, y_train_pred, y_test_pred):
    scores = DataFrame([], index=['train', 'test'])
    scores.loc['train', 'MAE'] = MAE(y_train, y_train_pred)
    scores.loc['train', 'MBE'] = MBE(y_train, y_train_pred)
    scores.loc['train', 'RMSE'] = RMSE(y_train, y_train_pred)
    scores.loc['train', 'MAPE'] = MAPE(y_train, y_train_pred)
    scores.loc['train', 'r2_score'] = r2_score(y_train, y_train_pred)
    scores.loc['test', 'MAE'] = MAE(y_test, y_test_pred)
    scores.loc['test', 'MBE'] = MBE(y_test, y_test_pred)
    scores.loc['test', 'RMSE'] = RMSE(y_test, y_test_pred)
    scores.loc['test', 'MAPE'] = MAPE(y_test, y_test_pred)
    scores.loc['test', 'r2_score'] = r2_score(y_test, y_test_pred)
    return scores
