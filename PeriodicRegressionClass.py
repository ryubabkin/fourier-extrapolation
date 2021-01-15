import pickle
import functions

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
        self._params = dotdict({'time_step' : time_step,
                                'top_n' : top_n,
                                'n_max' : n_max,
                                'cv' : cv,
                                'lags' : lags,
                                'lag_freq' : lag_freq,
                                'max_correction' : max_correction})
        self._utils = dotdict({})
        self.regressor = None
        self.spectrum = None
        self.scores = None
        self.__module__ = "PeriodicRegressionClass"
        return

    def fit(self, data,
            date_format = "%Y-%m-%d %H:%M:%S",
            ):
        if self._params.max_correction is None:
            self._params.max_correction = int(len(data)*0.1)

        data = functions.data_init(df = data,
                                   date_format = date_format)
        time_series, self._utils.dt_freq = functions.get_time_series(data)
        self._utils.dt_start = time_series['dt'].min()
        self._utils.dt_end = time_series['dt'].max()
        self._utils.time_series = functions.fill_missing(time_series)
        self._utils.trend, self._utils.polyvals = functions.get_trend(self._utils.time_series['y'])
        signal = self._utils.time_series['y'] - self._utils.trend

        if self._params.top_n is None:
            self._params.top_n, self._utils.n_mae_score = functions.define_optimal_n(signal = signal,
                                                                                     cv = self._params.cv,
                                                                                     n_max = self._params.n_max)
        else:
            self._utils.n_mae_score = None

        self._utils.correction_cut, self._utils.corrections = functions.find_length_correction(signal = signal,
                                                                                               max_correction = self._params.max_correction,
                                                                                               top_n = self._params.top_n,
                                                                                               cv = self._params.cv)

        self.spectrum = functions.get_frequencies(signal = signal[:-self._utils.correction_cut])
        prepared_data = functions.create_train_data(data = self._utils.time_series,
                                                    spectrum = self.spectrum,
                                                    top_n = self._params.top_n,
                                                    lags = self._params.lags,
                                                    lag_freq = self._params.lag_freq,
                                                    ).reset_index(drop = True)

        self.regressor, self.scores, self._utils.train_result = functions.train_regression(data = prepared_data,
                                                                                    cv = self._params.cv)



    def predict(self, start, end):
        pred_data = functions.create_predict_data(data = self._utils.time_series,
                                                  model = self.model,
                                                  start_date = start,
                                                  end_date = end,
                                                  freq = self._utils.dt_freq)
        return prediction

    def plot_spectrum(self, save_to = None, log = False):
        scale = len(self._utils.time_series) / (self._utils.dt_end - self._utils.dt_start).total_seconds() * 3600
        functions.plot_spectrum(spectrum = self.spectrum,
                                top_n = self._params.top_n,
                                save_to = save_to,
                                log = log,
                                scale = scale)

    def plot_train_results(self, x_lim = None, y_lim = None,save_to = None):
        functions.plot_train_results(results = self._utils.train_result,
                                     x_lim = x_lim,
                                     y_lim = y_lim,
                                     save_to = save_to)

    def plot_corrections(self, save_to = None):
        functions.plot_corrections(corrections = self._utils.corrections,
                                   save_to = save_to)

    def save_model(self, modelfile):
        f = open(modelfile, 'wb')
        pickle.dump(self, f, -1)
        f.close()

    def load_model(modelfile):
        f = open(modelfile, 'rb')
        model = pickle.load(f)
        f.close()
        return model
