# AI-InvestiBot
## This project is under development, but the changes since last year are not on github.

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

The project aims to be a flexable stock predictor using AI. This project is under development, but the changes since last year are not on github.

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
- **Lambda Version**: Allows the bot to be run without keeping a laptop open(It is also very, very cheap to use).
- **AI Techniques such as**:
  + Price AND Percentage Models
  + Data Augmentation
  + Transfer learning
  + Early Stopping
  + Etc

# Planned Additions

The following features are planned to be added in the future:

- [x] Getting the model to learn on the data.
- [x] Easy way to add many models using call backs
- [ ] Reach Library standards such as:
  - [ ] Bug Fixes
  - [ ] More Documentation
  - [ ] More Flexibility
  - [ ] More verification of a high accuracy rate.
- [x] Fix Issues added by PercentageModel Refactor


# How To Start

WARNING: It looks like the Model is currently off by a few days of something. Do NOT use to make money yet.
1) Get data using get_info.py
2) Train and save the models, look at the end of models.py for an example of how to do this. You have to train and save it yourself since I have removed everything in the Stocks folder.
3) Look at the current implementations in implementation.py.
4) Use them if you like them or add more if you want to customize it(lamda version does not work)

P.S: Remember to change the api key in secret key. 

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


# How to interpret

- You can have have confidence becuase:
  + The model has never seen the data
  + Not over fitted becuase Model used Early stopping
  + NO Transfer learning applied. Once transfer is applied, it will become even more accuracte(hopefully).
  + It has been tested on other similar stocks(on `PercentageModel` only) and has shown equally promising results

  * The only thing that may be wrong is that the model may accedently get future data.


- Directional Test is how often the predicted and test moved together.
  + Directional Test:  75.4345
  + Means 75% accuracy in the change in direction

- Spatial is what sees if the predicted is correctly positioned in relation to the real data. So if it goes up, the predicted should be over, but if it goes down, the predicted should go down
  + Spatial Test:  68.3548398
  + Means 68% accuracy if the % predicted is above if the % is positive and bellow if the A% is negative
- RMSE and RMSSE shows how incorrect the bot is. RMSSE is more impacted by larger differences. Remember that the lower the value of these metrics, the better the performance.
