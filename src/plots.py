from collections import OrderedDict
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)

from plotly.graph_objs import *

import sys
sys.path.append('./analysis.py')

from analysis import *

import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR

def plot_all_data(data):
    variables_order = ['Temperature', 'Dew Point', 'Humidity', 'Sea Level Press.', 'Visibility', 'Wind', 'Precipitation']
    variables_units = {
        'Temperature': '&deg;F',
        'Humidity': '%',
        'Dew Point': '&deg;F',
        'Precipitation': 'in',
        'Sea Level Press.': 'in',
        'Wind': 'mph',
        'Visibility': 'mi'
    }
    variables = {
        'Temperature': ['temp_f_low','temp_f_avg','temp_f_high'],
        'Humidity': ['humidity_%_low','humidity_%_avg','humidity_%_high'],
        'Dew Point': ['dew_point_f_low','dew_point_f_avg','dew_point_f_high'],
        'Precipitation': ['precip_in_sum'],
        'Sea Level Press.': ['sea_level_press_in_low','sea_level_press_in_avg','sea_level_press_in_high'],
        'Wind': ['wind_gust_mph_high','wind_mph_avg','wind_mph_high'],
        'Visibility': ['visibility_mi_low','visibility_mi_avg','visibility_mi_high']
    }
    
    traces = []

    for key in variables_order:
        if key == 'Precipitation':
            traces.append(Scatter(x=data['date'],
                                  y=data[variables[key][0]],
                                  name='Total',
                                  visible=False))
        else:
            traces.append(Scatter(x=data['date'],
                                  y=data[variables[key][0]],
                                  name='Gust' if key == 'Wind' else 'Low',
                                  visible = True if key == 'Temperature' else False))
            traces.append(Scatter(x=data['date'],
                                  y=data[variables[key][1]],
                                  name='Average',
                                  visible = True if key == 'Temperature' else False))
            traces.append(Scatter(x=data['date'],
                                  y=data[variables[key][2]],
                                  name='High',
                                  visible = True if key == 'Temperature' else False))
    
    updatemenus_buttons = []

    for i, key in enumerate(variables_order):
        vals = [False] * len(traces)

        if key == 'Precipitation':
            vals[i * 3] = True
        else:
            vals[i * 3] = True
            vals[i * 3 + 1] = True
            vals[i * 3 + 2] = True

        updatemenus_buttons.append({
            'args': [{'visible': vals},{'yaxis': {'title': '%s [%s]' % (key, variables_units[key])}}],
            'label': key,
            'method': 'update'
        })
        
    interactive_layout = Layout({
        'title': 'Weather History and Observations for San Luis Obispo, CA',
        'xaxis': {
            'rangeselector': {
                'buttons': [
                    {
                        'count': 1,
                        'label': '1m',
                        'step': 'month',
                        'stepmode': 'backward'
                    },
                    {
                        'count': 6,
                        'label': '6m',
                        'step': 'month',
                        'stepmode': 'backward'
                    },
                    {
                        'count': 1,
                        'label': 'YTD',
                        'step': 'year',
                        'stepmode': 'todate'
                    },
                    {
                        'count': 1,
                        'label': '1y',
                        'step': 'year',
                        'stepmode': 'backward'
                    },
                    {
                        'step': 'all'
                    }
                ]
            },
            'rangeslider': {},
            'type': 'date'
        },
        'yaxis': {'title': 'Temperature [&deg;F]'},
        'updatemenus': [{
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': True,
            'type': 'buttons',
            'x': 0,
            'xanchor': 'left',
            'y': -0.6,
            'yanchor': 'bottom',
            'buttons': updatemenus_buttons
        }]
    })
    
    interactive_fig = Figure(data=traces, layout=interactive_layout)
    py.iplot(interactive_fig)
    
def ml_plot_helper(x, traces, trace_name, title):
    data = []
    
    for i, trace in enumerate(traces):
        # prepare data for plot
        data.append(
            Scatter(x=np.array(x),
                    y=trace,
                    name=trace_name[i])
        )

    layout = Layout({
        'title': title
        }
    )
    
    # plot scores over persistence values
    fig = Figure(data=data, layout=layout)
    py.iplot(fig)
    
def simple_time_series_forecast_persistence_rmse(data):
    # split into train and test sets
    X = data['temp_f_low'].values
    train, test = X[0: -730], X[-730:]
    
    persistence_values = range(1, 731)
    scores = list()

    for p in persistence_values:
        # walk-forward validation
        history = [x for x in train]
        predictions = list()

        for i in range(len(test)):
            # make prediction
            yhat = history[-p]
            predictions.append(yhat)

            # observation
            history.append(test[i])

        # report performance
        rmse = np.sqrt(mean_squared_error(test, predictions))
        scores.append(rmse)
    
    # plot scores over persistence values
    ml_plot_helper(persistence_values, [scores], ['RMSE'], 'Persisted Observation to RMSE on the Dayli Temperature for San Luis Obispo, CA')
    
def simple_time_series_forecast_expanding_window_rmse(data):
    persistence_values = range(1, 731)
    
    # split into train and test sets
    X = data['temp_f_low'].values
    train, test = X[0: -730], X[-730:]

    rmse, predictions = calc_RMSE_for(train, test, np.mean)
    print('RMSE for mean: %.3f' % rmse)

    rmse, predictions = calc_RMSE_for(train, test, np.median)
    print('RMSE for median: %.3f' % rmse)
    
    # plot predictions vs observations
    ml_plot_helper(persistence_values, [test, predictions], ['Test', 'Forecast'], 'Line Plot of Predicted Values vs Test Dataset for the Median Expanding Window Model')
    
def simple_time_series_forecast_rolling_window_rmse(data):
    persistence_values = range(1, 731)
    
    # split into train and test sets
    X = data['temp_f_low'].values
    train, test = X[0: -730], X[-730:]
    
    scores = calc_RMSE_rolling_window_for(train, test, np.mean, 731)

    # plot predictions vs observations
    ml_plot_helper(persistence_values, [scores], ['RMSE'], 'Line Plot of Rolling Window Size to RMSE for a Mean Forecast on the Daily Temperature for San Luis Obispo, CA')
    
    scores = calc_RMSE_rolling_window_for(train, test, np.median, 731)

    # plot predictions vs observations
    ml_plot_helper(range(1, 731), [scores], ['RMSE'], 'Line Plot of Rolling Window Size to RMSE for a Median Forecast on the Daily Temperature for San Luis Obispo, CA')
    
def out_of_sample_forecast_arima(data):
    # split into train and test sets
    split_point = len(data) - 7
    data_train, data_test = data[0:split_point], data[split_point:]
    
    # seasonal difference
    X = data['temp_f_low'].values
    days_in_year = 365
    differenced = difference(X, days_in_year)
    
    # fit model
    model = ARIMA(differenced, order=(7,0,1))
    model_fit = model.fit(disp=0)
    
    # multi-step out-of-sample forecast
    forecast = model_fit.forecast(steps=7)[0]

    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1

    tests = []
    predictions = []

    for yhat in forecast:
        test = data_test.iloc[day - 1]['temp_f_low']
        predicted = inverse_difference(history, yhat, days_in_year)

        print('Day %d -- Forecast: %f, Actual: %f' % (day, predicted, test))
        history.append(predicted)
        day += 1

        tests.append(test)
        predictions.append(predicted)

    test_score = np.sqrt(mean_squared_error(tests, predictions))
    print('Test RMSE: %.3f' % test_score)
    
    # plot predictions vs observations
    ml_plot_helper(data[split_point:]["date"], [tests, predictions], ['Test', 'Forecast'], 'Line Plot of Predicted Values vs Test Dataset - San Luis Obispo, CA')
    
def fixed_autoregression_forecast(data):
    # split into train and test sets
    X = data['temp_f_low'].values

    train, test = X[1:len(X) - 7], X[len(X) - 7:]
    
    # train autoregression
    model = AR(train)
    model_fit = model.fit()

    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

    for i in range(len(predictions)):
        print('Day %d -- Forecast: %f, Actual: %f' % (i + 1, predictions[i], test[i]))

    error = np.sqrt(mean_squared_error(test, predictions))
    print('\nTest RMSE: %.3f' % error)
    
    # plot results
    ml_plot_helper(data[len(X) - 7:]["date"], [test, predictions], ['Test', 'Forecast'], 'Predictions From Fixed AR Model')
    
def rolling_autoregression_forecast(data):
    # split into train and test sets
    X = data['temp_f_low'].values

    train, test = X[1:len(X) - 7], X[len(X) - 7:]
    
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()

    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('Day %d -- Forecast: %f, Actual: %f' % (t + 1, yhat, obs))

    error = np.sqrt(mean_squared_error(test, predictions))
    print('\nTest RMSE: %.3f' % error)
    
    # plot
    ml_plot_helper(data[len(X) - 7:]["date"], [test, predictions], ['Test', 'Forecast'], 'Predictions From Rolling AR Model')