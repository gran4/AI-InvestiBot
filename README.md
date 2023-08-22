# AI-InvestiBot

## Table of Contents

- [Introduction](#introduction)
- [Contact Us](#contact-us)
- [Features](#features)
- [Planned Additions](#planned-additions)
- [How it works](#how-it-works)
  - [Information Retrieval and Caching](#information-retrieval-and-caching)
  - [Unique Indicators in Models](#unique-indicators-in-models)
  - [Stock Bot Functionality](#stock-bot-functionality)
  - [Bot Selection Process](#bot-selection-process)
  - [Earnings Processing](#earnings-processing)
- [Comparing Models](#comparing-models)
- [Additional Information](#additional-information)


# Introduction

This repository is currently under active development. The project aims to be more accurate than other projects by providing innovative features not often found in other stock bots.

# Contact Us
Discord: https://discord.gg/uHqBrqrr


# Features

- **Unique Indicators**: The project includes unique indicators, which can be found in the `get_info.py` file.
- **Non-Daily Indicators**: Unlike most bots that rely on daily indicators, this project incorporates indicators that are not limited to daily data, such as earnings.
- **Flexible Model Creation**: Users have the freedom to create their own models using the `information_keys` feature.
- **Ingenious Method for Custom Models**: A callback function can be passed to the `train` function(traning the model) that creates the custom model. This allows you do use LSTMs, CNNs, Tranformers, or any type of model you want
- **Ingenious Method for Adding New AI Models**: BaseModel is the base that basically handles all the non-ai related stock bot functionality. This allows for child classes that easily use different structures.
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
- **Active Development**: The project is actively being developed, with regular updates and improvements.


# Planned Additions

The following features are planned to be added in the future:

- [x] Achieving a 80% accuracy rate on previously untrained data.
- [x] Easy way to add many models using call backs
- [ ] Reach Library standards such as:
  - [ ] Bug Fixes
  - [ ] More Documentation
  - [ ] More Flexibility
  - [ ] More verification of the high accuracy rate.


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
- Percentage Model: This is the base child class that uses data scaled btw high and low of the window data and outputs the predicted % change in price
- Training, testing, saving, and loading are handled by separate functions(Ensuring quality ai).
- Training can be a test, using only the first 80% of data
- Information for each day is obtained through two methods:
  - Method 1: Offline (past data only)
    - Relies on data from `get_info.py`.
    - In this case, `model.cached_info` is always a dictionary or None.
  - Method 2: Online
    - Utilizes data from yfinance.
    - Once 280 days of past data are obtained, the oldest day is removed, and a new day is added at the end.
    - In this case, `model.cached_info` is always a pandas DataFrame or None.

## Bot Selection Process

The bot selects stocks based on the following criteria:

- The bot identifies the most promising stocks.
- It utilizes the available funds, following the rules set by the `ResourceManager` class.
- Stocks are held if their performance exceeds a certain threshold (`MAX_HOLD_INDEX`).
- Stocks are bought if specific conditions are met, including:
  - All models' profit ratios are above `PREDICTION_THRESHOLD`.
  - The average profit ratio exceeds the `RISK_REWARD_RATIO`.
- The lambda and loop implemenations use the same functions.
  - Therefore, more implementations can easily be added

## Earnings Processing

The project processes earnings in the following manner:

- All earnings are obtained and separated into two lists: dates and the difference between actual and estimated values.
- During runtime, earnings outside of a specific range are removed.
- The processed earnings are transformed into a continuous list:
  - Earnings are represented as 0 if no earnings occurred on a specific day.
  - The difference between the expected and actual values is used when earnings occur.
- Certain limitations prevent the stock bot from detecting earnings in some cases, which is an issue currently being addressed.




# RESULTS

This project offers various models to choose from, including:

- Base Model: This is the parent class for all other models and has no data of its own unless specified.
- Price Model: This is the base class that uses data scaled btw high and low of company data and outputs the predicted price
- Percentage Model: This is the base class that uses data scaled btw high and low of the window data and outputs the predicted % change in price

- Day Trade Model:
  - Directional Test: 96.53061224489797
  - Spatial Test: 92.44897959183673
  - Test RMSE: 1.2622329815597677
  - Test RMSSE: 55.46791004456621
- Impulse MACD Model:
  - Directional Test: 94.08163265306122
  - Spatial Test: 93.06122448979592
  - Test RMSE: 0.5951596806097224
  - Test RMSSE: 10.311187639192923
- Reversal Model:
  - Directional Test: 95.51020408163265
  - Spatial Test: 93.87755102040816
  - Test RMSE: 0.695032509478849
  - Test RMSSE: 14.778610864381438
- Earnings Model:
  - Directional Test: 95.51020408163265
  - Spatial Test: 91.42857142857143
  - Test RMSE: 0.7749913091267872
  - Test RMSSE: 34.070680956005674
- RSI Model:
  - Directional Test:  97.14285714285714
  - Spatial Test:  95.71428571428572
  - Test RMSE: 0.5837482545772584
  - Test RMSSE: 22.226485198086568
- Breakout Model:
  - Directional Test: 90.81632653061224
  - Spatial Test: 88.77551020408163
  - Test RMSE: 0.8159638992035801
  - Test RMSSE: 25.95683177058772
- Super Trends Model:
  - Directional Test: 93.26530612244898
  - Spatial Test: 90.40816326530611
  - Test RMSE: 1.5785158656115141
  - Test RMSSE: 121.95386912720913

# How to interperat

- YOU CAN HAVE CONFIDENCE BECUASE:
  + The model has never seen the data
  + Not over fitted becuase Model used Early stopping
  + NO Transfer learning applied. Once transfer is applied, it will become even more accuracte.
  + It has been tested on other similar stocks(on `PercentageModel` only) and has shown equally promising results
  + More methods for testing will be added soon

- Directional Test is how often the predicted and test moved together.
  + Directional Test:  93.26530612244898
  + Means 93% accuracy

- Spatial is what sees if the predicted is correctly positioned in relation to the real data. So if it goes up, the predicted should be over, but if it goes down, the predicted should go down
  + Spatial Test:  94.26530612244898
  + Means 94% accuracy

- RMSE and RMSSE show the amount it is off. RMSSE is more impacted by larger differences. Remember that the lower the value of these metrics, the better the performance.
