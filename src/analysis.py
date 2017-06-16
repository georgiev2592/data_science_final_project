from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

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

def predict_finals_week(data):
    dfs = []
    
    dfs.append(pd.DataFrame({"date": ["2017-06-12", "2017-06-13", "2017-06-14", "2017-06-15", "2017-06-16", "2017-06-17", "2017-06-18", "2017-06-19"]}))
    
    for ndx in data.columns:
        raw_data = {}
        
        # split into train and test sets
        X = data[ndx].values
        
        try:
            # train autoregression
            model = AR(X)
            model_fit = model.fit()
            window = model_fit.k_ar
            coef = model_fit.params

            raw_data[ndx] = []

            # make predictions
            history = X[len(X)-window:]
            history = [history[i] for i in range(len(history))]
            predictions = list()

            for t in range(8):
                length = len(history)
                lag = [history[i] for i in range(length-window,length)]
                yhat = coef[0]

                for d in range(window):
                    yhat += coef[d+1] * lag[window-d-1]

                predictions.append(yhat)
                history.append(yhat)

            for prediction in predictions:
                raw_data[ndx].append(prediction)

            dfs.append(pd.DataFrame(raw_data))
        except:
            continue
            
    return pd.concat(dfs, axis=1)