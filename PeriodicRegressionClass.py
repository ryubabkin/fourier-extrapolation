import pickle

import functions as f


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PeriodicRegression(object):

    def __init__(self,
                 time_step='1h',  # time series step
                 top_n=None,  # top n frequencies
                 n_max=20,  # max frequencies for top_n = "auto"
                 cv=0.1,  # validation part
                 lags=None,  # list of lag steps
                 lag_freq=None,  # time  step for lags
                 max_correction=None,  # max length for correction, 0.1 of length
                 fill_ranges=None  # [short fill range, long fill range] in str freqs
                 ):
        self._params = DotDict({'time_step': time_step,
                                'top_n': top_n,
                                'n_max': n_max,
                                'cv': cv,
                                'lags': lags,
                                'lag_freq': lag_freq,
                                'max_correction': max_correction,
                                'fill_ranges': fill_ranges})

        if (self._params.cv >= 1) or (self._params.cv < 0):
            self._params.cv = 0.1
        self._utils = DotDict({})
        self.regressor = None
        self.spectrum = None
        self.scores = None
        self.prepared_data = None
        self.__module__ = "PeriodicRegressionClass"
        return

    def fit(self, data,
            date_format="%Y-%m-%d %H:%M:%S",
            ):
        if self._params.max_correction is None:
            self._params.max_correction = int(len(data) * 0.1)

        data = f.data_init(df=data,
                           date_format=date_format)
        time_series, self._utils.dt_freq = f.get_time_series(df=data,
                                                             freq=self._params.time_step)
        self._utils.dt_start = time_series['dt'].min()
        self._utils.dt_end = time_series['dt'].max()
        self._utils.cv = int(self._params.cv * len(time_series))
        self._utils.time_series, self._utils.missing = f.fill_missing(data=time_series,
                                                                      freq=self._utils.dt_freq,
                                                                      ranges=self._params.fill_ranges)
        self._utils.trend, self._utils.polyvals = f.get_trend(self._utils.time_series['y'])
        signal = self._utils.time_series['y'] - self._utils.trend

        if self._params.top_n is None:
            self._params.top_n, self._utils.n_mae_score = f.define_optimal_n(signal=signal,
                                                                             cv=self._utils.cv,
                                                                             n_max=self._params.n_max)
        else:
            self._utils.n_mae_score = None
        self._utils.correction = f.find_length_correction(signal,self._params.max_correction, self._utils.cv)
        self.spectrum = f.get_frequencies(signal=signal[self._utils.correction:-self._utils.cv-1])
        self._utils.top_spectrum = f.correct_top_spectrum(signal, self._params.top_n, self._params.max_correction, self._utils.cv)

        self.prepared_data = f.create_train_data(data=self._utils.time_series,
                                                 spectrum=self._utils.top_spectrum,
                                                 top_n=self._params.top_n,
                                                 lags=self._params.lags,
                                                 lag_freq=self._params.lag_freq,
                                                 ).reset_index(drop=True)

        self.regressor, self.scores, self._utils.train_result = f.train_regression(data=self.prepared_data,
                                                                                   cv=self._utils.cv)

    def predict(self, start, end):
        prediction = f.create_predict_data(data=self._utils.time_series,
                                           start_date=start,
                                           end_date=end,
                                           freq=self._utils.dt_freq)
        return prediction

    def plot_spectrum(self, save_to=None, log=False):
        scale = len(self._utils.time_series) / (self._utils.dt_end - self._utils.dt_start).total_seconds() * 3600
        f.plot_spectrum(spectrum=self.spectrum,
                        top_n=self._params.top_n,
                        save_to=save_to,
                        log=log,
                        scale=scale)

    def plot_train_results(self, x_lim=None, y_lim=None, save_to=None):
        f.plot_train_results(results=self._utils.train_result,
                             x_lim=x_lim,
                             y_lim=y_lim,
                             save_to=save_to)

    def plot_missing_data(self, frame=None, save_to=None):
        f.plot_missing_data(data=self._utils.time_series,
                            missing=self._utils.missing,
                            freq=self._utils.dt_freq,
                            frame=frame,
                            save_to=save_to)

    def plot_corrections(self, save_to=None):
        f.plot_corrections(corrections=self._utils.corrections,
                           save_to=save_to)

    def save_model(self, modelfile):
        file = open(modelfile, 'wb')
        pickle.dump(self, file, -1)
        file.close()

    @staticmethod
    def load_model(modelfile):
        file = open(str(modelfile), 'rb')
        model = pickle.load(file)
        file.close()
        return model
