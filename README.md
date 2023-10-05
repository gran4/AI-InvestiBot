# AI-InvestiBot

## Table of Contents

- [Introduction](#introduction)
- [Contact Us](#contact-us)
- [Features](#features)
- [Planned Additions](#planned-additions)
- [How to start](#how-to-start)
- [How it works](#how-it-works)
  - [Information Retrieval and Caching](#information-retrieval-and-caching)
  - [Unique Indicators in Models](#unique-indicators-in-models)
  - [Stock Bot Functionality](#stock-bot-functionality)
  - [Bot Selection Process](#bot-selection-process)
  - [Earnings Processing](#earnings-processing)
- [Comparing Models](#comparing-models)
- [Additional Information](#additional-information)


# Introduction

This repository is currently under active development. The project aims to be more accurate than other projects by providing innovative features not often found in other stock bots; with cleaner and more modular code.

# Contact Us
Discord: https://dsc.gg/ai-investibot/  (Uses a dsc link in order to get a custom link)


# Features

- **Unique Indicators**: The project uses unique indicators.
- **Non-Daily Indicators**: Unlike most bots, AI-InvestiBot uses indicators that are not limited to daily data, such as earnings.
- **Flexible**:
    + **Flexible Model Creation** Users have the freedom to create their own models using the `information_keys` feature.
    + **Fexible AI**: A callback function can be passed to the `train` function(traning the model) that creates the custom model. This allows you do use any type of model you want
- **ResourceManager Class**: The `ResourceManager` class is implemented to manage and direct financial resources effectively.
- **Predictions for Multiple Companies**: This project offers predictions for multiple companies per day, rather than just one.
- **Holding Stocks**: The stock bot has the capability to hold stocks.
- **Lambda Version**: Allows the bot to be run without keeping a laptop open(It is also should be cheap to use).
- **AI Techniques such as**:
  + Multiple Models
  + Data Augmentation
  + Transfer learning
  + Early Stopping
  + Etc
- **Active Development Planned**: Taking a small break from the project.

# Planned Additions

The following features are planned to be added in the future:

- [x] Achieving a 80% accuracy rate on previously untrained data.
- [x] Easy way to add many models using call backs
- [ ] Reach Library standards such as:
  - [ ] Bug Fixes
  - [ ] More Documentation
  - [ ] More Flexibility
  - [ ] More verification of the high accuracy rate.
- [ ] Fix Issues added by PercentageModel Refactor


# How To Start

WARNING: The real time trading features need more testing. Do NOT use to make money yet.
 + Little code snippets at the bottom of each file, shows how to run it(in if __name__ == "__main__").
1) Get data using get_info.py
2) Train and save the models, look at the end of models.py for an example of how to do this. You have to train and save it yourself since I have removed everything in the Stocks folder.
3) Look at the current implementations in implementation.py.
4) Use them if you like them or add more if you want to customize it(lamda version does not work)

P.S: Remember to change the api and secret key in secrets.config. 

# How It Works

## Information Retrieval and Caching

The project retrieves and caches information in the following manner:

- The `get_info.py` file processes all data obtained from yfinance.
- The information is stored as a dictionary in a JSON file.
- The `information_keys` feature retrieves values from each key in the JSON.

## Unique Indicators in Models

The models in this project incorporate unique indicators as follows:

- Models utilize the `information_keys` attribute.
- These keys correspond to the names of indicators created from `get_info.py`.
- The model retrieves a dictionary from the JSON file and extracts the list associated with the key.
- Features in the form of NumPy arrays are then fed into the Sequential model.
- Use different Features by inputing a list of information_keys into either `PriceModel` or `PercentageModel`

## Stock Bot Functionality

The stock bot operates based on the following principles:


- The AI is implemented into the childclasses of `BaseModel`. 
- Base Model: This is the parent class for all other models and has no data of its own unless specified. Holds functionality for bot NOT AI.
- Price Model: This is the base child class that uses data scaled btw high and low of company data and outputs the predicted price
- Percentage Model: This is the base child class that uses data scaled btw high and low of a window of data(the past num days) and outputs the predicted % change in price
- Training, testing, saving, and loading are handled by separate functions(Ensuring quality code).
- Training can be a test, using only the first 80% of data
- Information for each day is obtained through two methods:
  - Method 1: Offline (past data only)
    - Relies on data from `get_info.py`.
    - In this case, `model.cached_info` is always a dictionary or None.
  - Method 2: Online
    - Utilizes data from yfinance.
    - Once 280 days of past data are obtained, the oldest day is removed, and a new day is added at the end.
    - In this case, `model.cached_info` is always a pandas DataFrame or None.

## How the Bot Runs

- The bot identifies the most promising stocks.
- It utilizes your available funds, following the rules set by the `ResourceManager` class.
- Stocks are held if their performance exceeds a certain threshold (`MAX_HOLD_INDEX`).
- Stocks are bought if specific conditions are met, including:
  - All models' profit ratios are above `PREDICTION_THRESHOLD`.
  - The average profit ratio exceeds the `RISK_REWARD_RATIO`.
- The lambda and loop implemenations use the same base functions.
  - Therefore, more implementations can easily be added

## Earnings Processing

The project processes earnings in the following manner:

- All earnings are obtained and separated into two lists: dates and the difference between actual and estimated values.
- During runtime, earnings outside of a specific range are removed.
- The processed earnings are transformed into a continuous list:
  - Earnings are represented as 0 if no earnings occurred on a specific day.
  - The difference between the expected and actual values is used when earnings occur.
- Certain limitations prevent the stock bot from detecting earnings in some cases, which is an issue currently being addressed.




# RESULTS(FOR Price Model only)

This project offers various models to choose from, including:

- Base Model: This is the parent class for all other models and has no data of its own unless specified.
- Price Model: This is the base class that uses data scaled btw high and low of company data and outputs the predicted price
- Percentage Model: This is the base class that uses data scaled btw high and low of the window data and outputs the predicted % change in price

- Day Trade Model:
  - Directional Test:  97.88732394366197
  - Spatial Test:  95.07042253521126

  - Test RMSE: 1.3360315740699096
  - Test RMSSE: 24.995202143966043
- Impulse MACD Model:
  - Directional Test:  96.47887323943662
  - Spatial Test:  95.07042253521126
  - Test RMSE: 0.6948929238336506
  - Test RMSSE: 7.995023009594582
- Reversal Model:
  - Directional Test:  97.1830985915493
  - Spatial Test:  95.07042253521126
  - Test RMSE: 1.1254591884267255
  - Test RMSSE: 24.42872924716995
- Earnings Model:
  - Directional Test:  98.59154929577466
  - Spatial Test:  96.47887323943662
  - Test RMSE: 0.8682655262847199
  - Test RMSSE: 15.578685178744083
- RSI Model:
  - Directional Test:  97.14285714285714
  - Spatial Test:  95.71428571428572
  - Test RMSE: 0.5837482545772584
  - Test RMSSE: 22.226485198086568
- Breakout Model:
  - Directional Test:  97.88732394366197
  - Spatial Test:  93.66197183098592
  - Test RMSE: 1.0865094554480963
  - Test RMSSE: 21.424078134818295
- Super Trends Model:
  - Directional Test:  97.88732394366197
  - Spatial Test:  92.25352112676056
  - Test RMSE: 1.6947722097944153
  - Test RMSSE: 78.60191098762428

model.test(show_graph=True)
![Figure_1](https://github.com/gran4/AI-InvestiBot/assets/80655391/0c205922-e6f4-4113-9d9c-1f3c890d1f81)



# How to interpret

- You can have have confidence becuase:
  + The model has never seen the data
  + Not over fitted becuase Model used Early stopping
  + NO Transfer learning applied. Once transfer is applied, it will become even more accuracte(hopefully).
  + It has been tested on other similar stocks(on `PercentageModel` only) and has shown equally promising results

  * The only thing that may be wrong is that the model may accedently get future data.


- Directional Test is how often the predicted and test moved together.
  + Directional Test:  93.26530612244898
  + Means 93% accuracy

- Spatial is what sees if the predicted is correctly positioned in relation to the real data. So if it goes up, the predicted should be over, but if it goes down, the predicted should go down
  + Spatial Test:  94.26530612244898
  + Means 94% accuracy(in the space)
- RMSE and RMSSE shows how incorrect the bot is. RMSSE is more impacted by larger differences. Remember that the lower the value of these metrics, the better the performance.
