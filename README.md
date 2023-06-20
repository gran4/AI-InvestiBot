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
      + Train RMSSE: 0.022040952764193555
      + Test RMSSE: 0.01761718200725566
      
      + Train RMSSE: 1.3980556690290633
      + Test RMSSE: 1.2747887486868177
    - MACD Model
      + Train RMSE: 0.02109060309388235
      + Test RMSE: 0.020949530939790666

      + Train RMSSE: 0.8495771704378083
      + Test RMSSE: 0.5597532219707358
    - Impulse MACD Model
      + Train RMSSE: 0.021071730433130226
      + Test RMSSE: 0.021409005906004214

      + Train RMSSE: 0.430748623349818
      + Test RMSSE: 0.3854770536908599
    - Reversal Model
      + Train RMSSE: 0.02087958554162695
      + Test RMSSE: 0.0192081585935407

      + Train RMSSE: 0.40873092330802707
      + Test RMSSE: 0.42745412377982367
    - Earnings Model
      + Train RMSSE: 0.023500793525848787
      + Test RMSSE: 0.01964422511707269

      + Train RMSSE: 0.6397388997572792
      + Test RMSSE: 0.6214688759571487
    - Breakout Model
      + Train RMSSE: 0.024133705747590935
      + Test RMSSE: 0.025299479271081993

      + Train RMSSE: 0.9638551303328443
      + Test RMSSE: 1.0401114858071763
  + Alot of information for you to choose from
    - earnings dates(processed in runtime)
    - earnings diffs(processed in runtime)
    - 12-day EMA
    - 26-day EMA
    - flips(bettween 12 and 26 day EMA)
    - 200-day EMA
    - MACD
    - Signal Line
    - Historgram
    - Change
    - Momentum
    - RSI
    - TRAMA
    - gradual-liqidity spike
    - 3-liqidity spike
    - momentum_oscillator

P.S:
  + RMSE stands for Root Mean Squared Error
    - It provides a single value that represents the average magnitude of the errors
    - Easily interpretable.
  + RMSSE stands for Root Mean Squared Scaled Error
    - It accounts for variations in the scale and magnitude of different stocks
    - Allows for more meaningful comparisons across different time series.
  + The lower the better.

