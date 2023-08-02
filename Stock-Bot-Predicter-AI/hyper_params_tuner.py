import json
import numpy as np

from trading_funcs import get_relavant_values, create_sequences, company_symbols
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber, CategoricalCrossentropy, KLDivergence, BinaryCrossentropy
from tensorflow.keras.activations import relu, elu, tanh, linear
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import ParameterGrid

from math import log

from models import *
# model = ImpulseMACDModel()
# model.train(epochs=20)
# model.load()
# model.test()
# raise ValueError()
def create_model(optimizer=Adam, loss=MeanSquaredError, activation_func=relu, neurons=64, learning_rate=0.001, num_days=60, information_keys=['Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']):
    optimizer = optimizer(learning_rate=learning_rate)
    loss = loss()
    # Build the LSTM model
    shape = len(information_keys)
    if 'earnings_dates' in information_keys:
        shape -= 1
    
    clear_session()
    model = Sequential()
    #model.add(LSTM(neurons, return_sequences=True, input_shape=(num_days, shape), activation=activation_func))
    #model.add(LSTM(neurons))
    model.add(LSTM(units=neurons, return_sequences=True, input_shape=(num_days, shape)))
    model.add(LSTM(units=neurons))
    model.add(Dense(units=1, activation=activation_func))
    model.compile(optimizer=optimizer, loss=loss)

    return model

def num_days_func1(days):
    return log(days / 60 + 1) * 60
def num_days_func2(days):
    return log((days + 30) / 20) * 40 + 10
def num_days_func3(days):
    return days / 100 + 40
def num_days_func4(days):
    return days / 60 + 50

param_grid = {
    'optimizer': [Adam],#
    'loss': [Huber],#, CustomLoss, CustomLoss2]
    'activation': [linear],#, relu],
    'neurons': [32],#24, 48, 64],# , 24, 98, 128
    'learning_rate': [.05], #, .005, 0.01, .05
    #'num_days_func': [num_days_func1, num_days_func2, num_days_func3, num_days_func4],
    'num_days': [100, 102, 104],
    'company': ['DIS'],#, 'NFLX', 'BRK-B', 'DIS', 'WMT'],
    'batch_size': [24]#, 32, 48]# 64
}
#[(3, 'TLSA'), (4, 'META'), (4, 'V'), (5, 'GOOG'), (5, 'NFLX'),
# (6, 'NVDA'), (7, 'AMZN'), (7, 'BRK-B'), (8, 'SBUX'),
# (9, 'ADBE'), (10, 'MSFT'), (10, 'T'), (11, 'AAPL'), (11, 'HD'), (11, 'NKE'), (13, 'CVS'), (13, 'WMT'), (15, 'MCD'), (16, 'DIS'), (16, 'BA'), (16, 'KO'), (16, 'IBM'), (16, 'XOM'), (25, 'GE')]

#3: 38 : TLSA
#4: 40 : META
#7: 58 : AMZN
#9: 74 : ADBE
#11: 90 : APPL
#13: 96 : CVS
#16: 102 : BA
#25: 103 : GE

param_grid2 = {
    'optimizer': [Adam, Adadelta],#
    'loss': [Huber, CustomLoss, CustomLoss2],#
    'activation': [linear, relu],
    'neurons': [24, 48, 64, 96],# , 24, 98, 128
    'learning_rate': [.025, .05, .1, .5], #, .005, 0.01, .05
    'num_days': [60, 100, 160],
    'batch_size': [16, 24, 32, 48]# 64
}

#{'activation': <function linear at 0x283111260>, 'batch_size': 16, 'learning_rate': 0.005, 'loss': <class 'keras.src.losses.Huber'>, 'neurons': 64, 'num_days': 100, 'optimizer': <class 'keras.src.optimizers.legacy.adam.Adam'>}
# 0.050756479315522296
# 59.83471074380166

#{'activation': <function relu at 0x283110720>, 'batch_size': 24, 'learning_rate': 0.01, 'loss': <class 'models.CustomLoss2'>, 'neurons': 48, 'num_days': 160, 'optimizer': <class 'keras.src.optimizers.legacy.adam.Adam'>}
# 0.05168317730971394
# 59.44954128440367
information_keys = ['Close', 'Histogram', 'Momentum', 'Change', 'ema_flips', 'signal_flips', '200-day EMA']

# Define early stopping criteria
early_stopping = EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')

hyper_params = []
rmsses = []
percents = []
models = []
for params in ParameterGrid(param_grid):
    company = params['company']
    optimizer = params['optimizer']
    loss = params['loss']
    activation = params['activation']
    neurons = params['neurons']
    learning_rate = params['learning_rate']
    #num_days_func = params['num_days_func']
    num_days = params['num_days']
    batch_size = params['batch_size']


    # Get relevant data
    _, data, _ = get_relavant_values(company, information_keys)
    shape = data.shape[1]

    #NOTE: Gets the relevant years
    with open(f'Stocks/{company}/dynamic_tuning.json', 'r') as file:
        dynamic_tuning = json.load(file)
    #data = data[-int(dynamic_tuning['relevant_years']*365):]
    #num_days = int(num_days_func(dynamic_tuning['relevant_years']*365))

    print(len(set(arr.dtype for arr in data.flatten())) == 1)
    print(dynamic_tuning['relevant_years'])
    print(num_days)

    # Process Data for LSTM
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size-num_days:]

    x_total, y_total = create_sequences(train_data, num_days)
    x_train, y_train = create_sequences(train_data, num_days)
    x_test, y_test = create_sequences(test_data, num_days)

    model = create_model(optimizer=optimizer, loss=loss, activation_func=activation, neurons=neurons, learning_rate=learning_rate, num_days=num_days)
    model.fit(x_train, y_train, callbacks=[early_stopping])

    def calculate_percentage_movement_together(list1, list2):
        total = len(list1)
        count_same_direction = 0
        count_same_space = 0

        for i in range(1, total):
            if (list1[i] > list1[i - 1] and list2[i] > list2[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list2[i - 1]):
                count_same_direction += 1
            if (list1[i] > list1[i - 1] and list2[i] > list1[i - 1]) or (list1[i] < list1[i - 1] and list2[i] < list1[i - 1]):
                count_same_space += 1

        percentage = (count_same_direction / (total - 1)) * 100
        percentage2 = (count_same_space / (total - 1)) * 100
        return percentage, percentage2
    y_pred = model.predict(x_test)
    temp = calculate_percentage_movement_together(y_test, y_pred)
    #if temp[0] < 48 or temp[1] < 48:
    #    continue
    percents.append(temp)

    params['num_days'] = num_days
    hyper_params.append(params)

    rmsse = np.sqrt(mean_squared_error(y_test, y_pred)) / np.mean(x_test)
    rmsses.append(rmsse)
    models.append(model)


temp = list(zip(hyper_params, rmsses, percents))
temp = sorted(temp, key=lambda x: x[2])
for params, rmsse, percent in temp:
    print(params)
    print(rmsse)
    print(percent)
    print()
#NOTE: Bellow is for top 3 results
# , reverse=True)
# for params, rmsse, percent in temp[:3]:
#     print(params)
#     print(rmsse)
#     print(percent)
#     print()
# print("DJEJDEJNEDNDENDENJDEJNDENJDEJNEDNJDENJDEKMDEKMMKSMKSWMKSWMKWSNJSWNJ")


# import matplotlib.pyplot as plt
# days_train = [i for i in range(len(y_pred))]

# x_test, y_test = create_sequences(test_data, params['num_days'])
# y_pred = temp[0][3].pred(x_test)

# predicted_test = plt.plot(days_train, y_pred, label='Predicted Test')
# actual_test = plt.plot(days_train, y_test, label='Actual Test')

# plt.title(f'{stock_symbol} Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend(
#     [predicted_test[0], actual_test[0]],#[real_data, actual_test[0], actual_train],
#     ['Predicted Test', 'Actual Test']#['Real Data', 'Actual Test', 'Actual Train']
# )
# plt.show()


