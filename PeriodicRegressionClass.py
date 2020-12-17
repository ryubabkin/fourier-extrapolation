
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            date_format = "%Y-%m-%d %H:%M:%S"):
        data = functions.data_init(df = data,
                         date_format = date_format
                         )
        self.time_series, self._dt_freq = functions.get_time_series(data)
        self._dt_start = self.time_series['dt'].min()
        self._dt_end = self.time_series['dt'].max()
        self.time_series = functions.fill_missing(self.time_series)
        self.time_series = functions.add_datetime_features(self.time_series)
        self._trend, self._polyvals = functions.get_trend(self.time_series['y'])
        
        signal = self.time_series['y'] - self._trend

        if top_n == 'auto':
            self._top_n, self._mae_score = functions.define_optimal_n(signal = signal,
                                                                      cv = cv,
                                                                      n_max = n_max
                                                                      )
        else:
            self._top_n = int(top_n)
            self._mae_score = None

        self.spectrum = functions.get_frequencies(signal = signal)
        self._length = len(signal)

    def predict(self, start, end, steps = None):
        restored = functions.restore_data(self, X_array)
        return restored

    def plot_spectrum(self, save_to = None, log = False):
        scale = self._length / (self._dt_end - self._dt_start).total_seconds() * 3600
        functions.plot_spectrum(spectrum = self.spectrum,
                                 top_n = self._top_n,
                                 save_to = save_to,
                                 log = log,
                                 scale = scale
                                 )
