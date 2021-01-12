
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import functions

class PeriodicRegression(object):
    def __init__(self,
                 time_step = '1h',                  # time series step
                 top_n = None,                      # top n frequencies
                 n_max = 20,                        # max freqs for top_n = "auto"
                 cv = 0.1,                          # validation part
                 lags = None,                       # list of lag steps
                 lag_freq = None,                   # timestep for lags
                 max_correction = None              # max length for correction, 0.1 of lenght
                 ):
        self.time_step = time_step
        self.top_n = top_n
        self.n_max = n_max
        self.cv = cv
        self.lags = lags
        self.lag_freq = lag_freq
        self.max_correction = max_correction
        return

    def fit(self, data,
            date_format = "%Y-%m-%d %H:%M:%S",
            ):
        if self.max_correction is None:
            self.max_correction = int(len(data)*0.1)

        data = functions.data_init(df = data,
                                   date_format = date_format)
        time_series, self.dt_freq = functions.get_time_series(data)
        self.dt_start = time_series['dt'].min()
        self.dt_end = time_series['dt'].max()
        self.time_series = functions.fill_missing(time_series)
        self.trend, self.polyvals = functions.get_trend(self.time_series['y'])
        signal = self.time_series['y'] - self.trend

        if self.top_n is None:
            self.top_n, self.n_mae_score = functions.define_optimal_n(signal = signal,
                                                                      cv = self.cv,
                                                                      n_max = self.n_max)
        else:
            self.n_mae_score = None

        self.cut, self.corrections = functions.find_length_correction(signal = signal,
                                                                      max_correction = self.max_correction,
                                                                      top_n = self.top_n,
                                                                      cv = self.cv)

        self.spectrum = functions.get_frequencies(signal = signal[:-self.cut])
        prepared_data = functions.create_train_data(data = self.time_series,
                                                    spectrum = self.spectrum,
                                                    top_n = self.top_n,
                                                    lags = self.lags,
                                                    lag_freq = self.lag_freq,
                                                    ).reset_index(drop = True)

        self.regressor, self.scores, self.train_result = functions.train_regression(data = prepared_data,
                                                                                    cv = self.cv)



    def predict(self, start, end):
        pred_data = functions.create_predict_data(data = self.time_series,
                                                  model = self.model,
                                                  start_date = start,
                                                  end_date = end,
                                                  freq = self.dt_freq)
        return prediction

    def plot_spectrum(self, save_to = None, log = False):
        scale = len(self.time_series) / (self.dt_end - self.dt_start).total_seconds() * 3600
        functions.plot_spectrum(spectrum = self.spectrum,
                                 top_n = self.top_n,
                                 save_to = save_to,
                                 log = log,
                                 scale = scale)

    def plot_train_results(self, x_lim = None, y_lim = None,save_to = None):
        functions.plot_train_results(results = self.train_result,
                                     x_lim = x_lim,
                                     y_lim = y_lim,
                                     save_to = save_to)

    def plot_corrections(self, save_to = None):
        functions.plot_corrections(corrections = self.corrections,
                                   save_to = save_to)
