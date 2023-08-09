import json
from tensorflow.keras.models import load_model
from trading_funcs import company_symbols


def get_years():
    lis = []
    for company in company_symbols:
        with open(f'Stocks/{company}/dynamic_tuning.json', 'r') as file:
            dynamic_tuning = json.load(file)
        lis.append((dynamic_tuning['relevant_years'], company))
    lis = sorted(lis, key=lambda x: x[0])

def get_transfer_learning_model():
    return load_model(f"transfer_learning_model")

def get_model(company):
    return load_model(f"Stocks/{company}")
