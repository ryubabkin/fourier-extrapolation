
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import functions

class PeriodicRegression(object):
    def __init__(self,
                 time_step = '1h',
                 n_freqs = 10):
        return

    def fit(self, data,
            top_n = 'auto',
            n_max = 20,
            cv = 0.1,
            date_format = "%Y-%m-%d %H:%M:%S",
            lags = None,
            lag_freq = None,
            max_correction = None):

        if max_correction is None:
            max_correction = int(len(data)*0.1)

        data = functions.data_init(df = data,
                         date_format = date_format)
        time_series, self._dt_freq = functions.get_time_series(data)
        self._dt_start = time_series['dt'].min()
        self._dt_end = time_series['dt'].max()
        self._time_series = functions.fill_missing(time_series)
        self._trend, self._polyvals = functions.get_trend(self._time_series['y'])
        self._length = len(self._time_series)

        signal = self._time_series['y'] - self._trend

        if top_n == 'auto':
            self._top_n, self._mae_score = functions.define_optimal_n(signal = signal,
                                                                      cv = cv,
                                                                      n_max = n_max)
        else:
            self._top_n = int(top_n)
            self._mae_score = None

        self._cut, self._corrections = functions.find_length_correction(signal = signal,
                                                                                  max_correction = max_correction,
                                                                                  top_n = self._top_n,
                                                                                  cv = cv)

        self._spectrum = functions.get_frequencies(signal = signal[:-self._cut])
        prepared_data = functions.create_train_data(data = self._time_series,
                                                    spectrum = self._spectrum,
                                                    top_n = self._top_n,
                                                    lags = lags,
                                                    lag_freq = lag_freq,
                                                    ).reset_index(drop = True)

        self._regressor, self._scores, self._train_result = functions.train_regression(data = prepared_data,
                                                                                    cv = cv)



    def predict(self, start, end):
        pred_data = functions.create_predict_data(data = self._time_series,
                                                  model = self.model,
                                                  start_date = start,
                                                  end_date = end,
                                                  freq = self._dt_freq)
        return prediction

    def plot_spectrum(self, save_to = None, log = False):
        scale = self._length / (self._dt_end - self._dt_start).total_seconds() * 3600
        functions.plot_spectrum(spectrum = self._spectrum,
                                 top_n = self._top_n,
                                 save_to = save_to,
                                 log = log,
                                 scale = scale)

    def plot_train_results(self, x_lim = None, y_lim = None,save_to = None):
        functions.plot_train_results(results = self._train_result,
                                     x_lim = x_lim,
                                     y_lim = y_lim,
                                     save_to = save_to)

    def plot_corrections(self, save_to = None):
        functions.plot_corrections(corrections = self._corrections,
                                   save_to = save_to)
