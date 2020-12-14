import pandas as pd
import numpy as np
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

def trend_feature(T, dates, deg = 2):
    polyvals = np.polyfit(x = np.arange(len(T)),
                          y = np.array(T.values),
                          deg = deg
                          )
    F = np.poly1d(polyvals.flatten())
    T_poly = pd.Series(F(np.arange(len(T))))
    T_poly.index = dates
    polyvals = polyvals.flatten().tolist()
    return(T_poly, polyvals)

def periodic_feature(HF, dates):
    Ampl = 0
    X = np.arange(0,len(dates))
    if len(HF)!=0:
        for freq in HF['freq']:
            A_sine = np.sin(2*np.pi*freq*X)
            A_cosine = np.cos(2*np.pi*freq*X)
            Ampl += (1j*A_sine + A_cosine)*complex(HF[HF['freq']==freq]['int'].values[0])
        Ampl = np.real(Ampl) / len(dates)*2
    else:
        Ampl = np.zeros(len(dates))
    Ampl = pd.Series(Ampl)
    Ampl.index = dates
    return(Ampl)


def MAPE(E,F,C):
    sigma = np.sum(abs(F - E)/C/4)/len(F)
    return sigma
def MAE(E,F):
    sigma = np.sum(abs(F - E))/len(F)
    return sigma
def MABE(E,F):
    sigma = np.sum(F - E)/len(F)
    return sigma
