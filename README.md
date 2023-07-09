# Stock-Bot-Predicter-AI

In dev stages. Stock Bots not completed yet.
Not even a stock bot yet, only has models(coming soon).


Currently working on:
  - starting to work on actual bot/automation

Plans to add:
  - Being able to run with loop
  - Being able to run with lambda

Thinking/Debating about:
  - loss functions being in Tradingfuncs.py


Currently impletmented:
  + Easy system for adding models
    - Models work with an information_keys attribute
      + These information_keys are the names of things created from the get_info.py
      + It gets a dict from a json and gets the list from the key
      + np.array of features are put into the Sequential model.
    - Base Model(Whatever you want)
      + It is the parent class for all other models
      + It has no data on its own unless specified
    - Day Trade Model 
      + Train RMSE: 0.02075242037941444
      + Test RMSE: 0.026771371361829204

      + Train RMSSE: 1.3539866279934776
      + Test RMSSE: 1.3284154118165579
    - MACD Model
      + Train RMSE: 0.023183201376227626
      + Test RMSE: 0.02732382064489737

      + Train RMSSE: 0.9503971603872557
      + Test RMSSE: 0.8466967848854924
    - Impulse MACD Model
      + Train RMSE: 0.021000604015479343
      + Test RMSE: 0.03453818087008323

      + Train RMSSE: 0.36591660531622144
      + Test RMSSE: 0.5035604190480721
    - Reversal Model
      + Train RMSE: 0.021183699586844264
      + Test RMSE: 0.028946277773079763

      + Train RMSSE: 0.42088951956685866
      + Test RMSSE: 0.5215137360642763
    - Earnings Model
      + Train RMSE: 0.023398385228929293
      + Test RMSE: 0.029933115015020682

      + Train RMSSE: 0.6544155735754877
      + Test RMSSE: 0.6228371371469489
    - Breakout Model
      + Train RMSE: 0.02172262269810893
      + Test RMSE: 0.028400576328573523

      + Train RMSSE: 0.8791136390349061
      + Test RMSSE: 1.0664367975707776
  + Alot of information for you to choose from
    - earnings dates(processed in runtime)
    - earnings diffs(processed in runtime)
    - 12-day EMA
    - 26-day EMA
    - ema_flips(bettween 12 and 26 day EMA)
    - signal_flips(bettween MACD and Signal line)
    - 200-day EMA
    - MACD
    - Signal Line
    - Historgram
    - Change
    - Momentum
    - RSI
    - TRAMA
    - Bollinger Middle
    - Bollinger Upper
    - Bolliner Lower
    - gradual-liqidity spike
    - 3-liqidity spike
    - momentum_oscillator
    - supertrend1
    - supertrend2
    - supertrend3
    - kumo_cloud

P.S:
  + Model trained for 100 epochs
  + RMSE stands for Root Mean Squared Error
    - Shows the absolute errors(How far away the prediction is from the expected)
    - It provides a single value that represents the average magnitude of the errors
    - Easily interpretable.
  + RMSSE stands for Root Mean Squared Scaled Error
    - It accounts for variations in the scale and magnitude of different stocks
    - Allows for more meaningful comparisons across different time series.
    - more reliable evaluation metric(a bot with a higher RMSE is better then a bot with higher RMSSE)
  + The lower the better.
  + Not trained on train data
