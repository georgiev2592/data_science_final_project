from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

def calc_RMSE_for(train, test, func):
    # walk-forward validation
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        # make prediction
        yhat = func(history)
        predictions.append(yhat)

        # observation
        history.append(test[i])

    # return performance
    return np.sqrt(mean_squared_error(test, predictions)), predictions

def calc_RMSE_rolling_window_for(train, test, func, max_window):
    window_sizes = range(1, max_window)
    scores = list()

    for w in window_sizes:
        # walk-forward validation
        history = [x for x in train]
        predictions = list()

        for i in range(len(test)):
            # make prediction
            yhat = func(history[-w:])
            predictions.append(yhat)

            # observation
            history.append(test[i])

        # report performance
        rmse = np.sqrt(mean_squared_error(test, predictions))
        scores.append(rmse)
        # print('w=%d RMSE:%.3f' % (w, rmse))
    
    return scores

def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]