# Stock-Bot-Predicter-AI

In dev stages. Stock Bots not completed yet.
Just added bot for loops

Features other stock bots do not have(Already added):
  - Unique indicators(look at get_info.py)
  - Indicators that are not daily(EX. earnings)
  - being able to create any model one wishes(using `information_keys`)
  - A `ResourceManager` class to limit/direct money
  - Predictions for many companies per day, not just one
  - HOLDING stocks
  - Active devolopement
  - A lambda version so you can run it without having a laptop open(COMING REALLY SOON)




Currently working on:
  - starting to work on actual bot/automation

Plans to add:
  - Being able to run with loop
  - Being able to run with lambda

Thinking/Debating about:
  - unique loss functions being in Tradingfuncs.py
  - unique times for each company(tedious, EX. tesla is in business for a shorter time then GE)



How it works:
  - How information is gotten and cached:
      + `get_info.py` processes all data gotten from yfinance.
      + The info is put as a Dict into a json
      + This is HOW `information_keys` works, it simply gets the value from each key
  - How unique indicators in Models:
      + Models work with an information_keys attribute
      + These information_keys are the names of things created from the get_info.py
      + It gets a dict from a json and gets the list from the key
      + np.array of features are put into the Sequential model.
  - How bot works:
      + training, testing, saving, and loadsing are SEPERATE functions
      + How it gets information for each day(2 methods):
        - Method 1 OFFLINE(PAST ONLY)
          + based off info from `get_info.py`
          + In this case, model.cached_info is always a Dict or None
        - Method 2 Online
          + bases it off of data from yfinance
          + Once past 280 days are gotten from yfinance, it updates by removing the first day and adding a new day to the end
          + In this case, model.cached_info is always a pd.DataFrame or None
      + How it selects for the bot:
        - The Bot selects for the most promising ones
        - Then uses your money, conforming to the `ResourceManager`
        - Holds if it is above `MAX_HOLD_INDEX`
        - Buys if all conditions are met:
          + All models profit ratio is above `PREDICTION_THRESHOLD`
          + Average is more then `RISK_REWARD_RATIO`
  - How non-daily indicators work.
      + They are excluded from normal processing. Special processing has to be applied(EX. earnings [↓Bellow↓↓↓↓↓↓↓↓]).
  - How earings works:
      + Starts by getting all the earnings and splitting them into
              the dates, and the difference bettween the actual and the estimated
              in 2 seperate lists
      + In runtime, it removes any earnings that are out of range
      + Then it processes it so it turns into 1 continues list.
        - It is 0 if no earnings, and the difference bettween excepted and actual if there is an earnings
        - There are certian points where the stock bot can not see earnings, that is an issue that needs to be fixedd



Comparing models:
    - Base Model(Whatever you want)
      + It is the parent class for all other models
      + It has no data on its own unless specified
    - Day Trade Model 
      + Train RMSE: 0.02075242037941444
      + Test RMSE: 0.026771371361829204

      + Train RMSSE: 1.3539866279934776
      + Test RMSSE: 1.3284154118165579
    - MACD Model
      + Train RMSSE: 0.9503971603872557
      + Test RMSSE: 0.8466967848854924
      + Train RMSSE: 0.9503971603872557
      + Test RMSSE: 0.8466967848854924
    - Impulse MACD Model
      + Train RMSSE: 0.36591660531622144
      + Test RMSSE: 0.5035604190480721
      + Train RMSSE: 0.36591660531622144
      + Test RMSSE: 0.5035604190480721
    - Reversal Model
      + Train RMSSE: 0.42088951956685866
      + Test RMSSE: 0.5215137360642763
      + Train RMSSE: 0.42088951956685866
      + Test RMSSE: 0.5215137360642763
    - Earnings Model
      + Train RMSE: 0.023398385228929293
      + Test RMSE: 0.029933115015020682

      + Train RMSSE: 0.6544155735754877
      + Test RMSSE: 0.6228371371469489
    - RSI Model
      + Train RMSSE: 0.8791136390349061
      + Test RMSSE: 1.0664367975707776
      + Train RMSSE: 0.8791136390349061
      + Test RMSSE: 1.0664367975707776
    - Breakout Model
      + Train RMSE: 0.025379195809305734
      + Test RMSE: 0.030050545088518107

      + Train RMSSE: 0.6776019152138987
      + Test RMSSE: 0.8600297293130289
  + Alot of information for you to choose from
    - earnings dates(processed in runtime)
    - earning diffs
    (processed in runtime)
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
    - Shows the absolute errors(How far away the prediction is from the expected)
    - It provides a single value that represents the average magnitude of the errors
    - Easily interpretable.
  + RMSSE stands for Root Mean Squared Scaled Error
    - It accounts for variations in the scale and magnitude of different stocks
    - Allows for more meaningful comparisons across different time series.
    - more reliable evaluation metric(a bot with a higher RMSE is better then a bot with higher RMSSE)
    - more reliable evaluation metric(a bot with a higher RMSE is better then a bot with higher RMSSE)
  + The lower the better.
  + Not trained on train data
