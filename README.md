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
    - Base Model(Whatever you want)
      + It is the parent class for all other models
      + It has no data on its own unless specified
    - Day Trade Model 
      + Train RMSSE: 27.992971353011118
      + Test RMSSE: 8.296044664633495
    - MACD Model
      + Train RMSSE: 27.46306710507532
      + Test RMSSE: 8.520355736058766
    - Impulse MACD Model
      + Train RMSSE: 20.360184344580677
      + Test RMSSE: 6.269214832919357
    - Reversal Model
      + Train RMSSE: 30.130229844560294
      + Test RMSSE: 7.708133225692995
    - Earnings Model
      + Train RMSSE: 26.418529628219893
      + Test RMSSE: 7.203620865873368
    - Breakout Model
      + Train RMSSE: 27.457170999083573
      + Test RMSSE: 6.881920637930147
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
  + RMSSE stands for Root Mean Squared Scaled Error
    - The lower the better.

