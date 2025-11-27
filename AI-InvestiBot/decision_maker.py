import json
from pandas_market_calendars import get_calendar
from models import load_models, break_out_indicators, ImpulseMACD_indicators, Reversal_indicators, RSI_indicators, super_trends_indicators
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


trading_calendar = get_calendar('XNYS')


def save_data_for_predictions(company_models, start_date, total_info_keys):
    predictions = []
    for model in company_models:
        predictions.append([])
    initial_date = pd.Timestamp(start_date, tz='America/Los_Angeles')
    initial_date = initial_date.tz_convert('UTC')
    new_date = trading_calendar.valid_days(start_date=initial_date, end_date=initial_date + pd.DateOffset(days=14))[-1]
    # Define the comparison date (2023-10-11 in this case)
    comparison_date = pd.Timestamp("2023-10-11", tz='America/Los_Angeles')
    # Check if the new date is past the comparison date
    assert new_date.tzinfo == initial_date.tzinfo
    
    first_model = company_models[0]
    cached_info = first_model.update_cached_info_online()
    cached = first_model.indicators_past_num_days(
        first_model.stock_symbol, start_date,
        total_info_keys, first_model.scaler_data,
        cached_info, first_model.num_days
    )
    while new_date < comparison_date:
        i = 0
        for model in company_models:
            processed_data = model.process_cached(cached)
            temp = model.predict(info=processed_data).flatten()
            temp = temp[::-1].tolist()
            predictions[i] += temp
            i += 1
        new_date = trading_calendar.valid_days(start_date=new_date, end_date=new_date + pd.DateOffset(days=14))[-1]
    return predictions


with open("secrets.config","rb") as f:
    company_information = json.load(f)


def train_decision_tree_classifier(features, labels, test_size=0.2, random_state=42):
    """Train and evaluate a basic decision tree on arbitrary features."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    return clf


def predict_company(decision_tree, feature_vector):
    """Return the predicted class (e.g., company index) for a single feature vector."""
    prediction = decision_tree.predict([feature_vector])[0]
    return prediction


def train_decision_maker():
    models, total_info_keys = load_models(strategys=[break_out_indicators, ImpulseMACD_indicators, Reversal_indicators, RSI_indicators, super_trends_indicators], names=['breakout', 'ImpulseMACD', 'Reversal', 'RSI', 'supertrends'])
    data = {}
    for company_models in models:
        print(type(company_models[0]))
        data[company_models[0].stock_symbol] = save_data_for_predictions(company_models, "2015-01-01", total_info_keys)
    with open(f"Stocks/data_for_decision_tree.json", "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    train_decision_maker()

