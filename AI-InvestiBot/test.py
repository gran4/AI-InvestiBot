from models import *

def test_many(model_class: BaseModel=ImpulseMACDModel, tests: int=20, *args, **kwargs):
    averages = []
    for i in range(tests):
        model = model_class()
        model.train(*args, **kwargs)
        if 'time_shift' in kwargs:
            results = model.test(time_shift = kwargs['time_shift'])
        else:
            results = model.test()
        averages.append(results)
    num_columns = len(averages[0])
    column_sums = [0] * num_columns
    for row in averages:
        for i, value in enumerate(row):
            column_sums[i] += value

    return [sum / len(averages) for sum in column_sums]


def test_individual_indepth(model_class: BaseModel=ImpulseMACDModel, ):
    model = model_class()
    from datetime import datetime, date
    from dateutil.relativedelta import relativedelta
    import json
    with open(f"Stocks/{model.stock_symbol}/info.json", 'r') as file:
        cached_info = json.load(file)
    with open(f'Stocks/{model.stock_symbol}/dynamic_tuning.json', 'r') as file:
        relevant_years = json.load(file)['relevant_years']
    model.start_date = datetime.strptime(cached_info['Dates'][-1], "%Y-%m-%d")-relativedelta(years=relevant_years)
    model.end_date = model.start_date+relativedelta(days=200+model.num_days)
    model.update_dates(model.start_date, model.end_date)
    model.load()
    print(model.start_date, model.end_date)

    money_made = 0
    stocks_hold = 0
    bought_at = []

    signals = {}
    time_stamp = 0
    while True:
        if model.update_cached_offline():
            break
        
        print(len(model.cached))
        prev_close = float(model.cached[-1][0])
        temp = model.predict(info=model.cached)[0][0]
        profit = float(temp/prev_close)
        if profit >= 1.01:
            signals[time_stamp] = 1
            if stocks_hold:
                bought_at.append(prev_close)
                stocks_hold += 1
        elif profit <= .98:
            if stocks_hold:
                for stock in bought_at:
                    money_made += prev_close-stock
                stocks_hold = 0
                bought_at = []
            signals[time_stamp] = -1
    print(money_made)
test_individual_indepth()
        
