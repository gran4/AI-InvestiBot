from trading_funcs import company_symbols
import json

lis = []
for company in company_symbols:
    with open(f'Stocks/{company}/dynamic_tuning.json', 'r') as file:
        dynamic_tuning = json.load(file)
    lis.append((dynamic_tuning['relevant_years'], company))
lis = sorted(lis, key=lambda x: x[0])
print(lis)