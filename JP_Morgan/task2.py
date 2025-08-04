from task1 import estimate_price
import pandas as pd
from datetime import datetime

def price_storage_contract(
    injection_dates,           
    inject_amounts,
    withdrawal_dates,
    withdrawal_amounts,
    injection_withdrawal_rate,
    max_capacity,             
    storage_cost_per_day 
):

    if sum(inject_amounts) != sum(withdrawal_amounts):
        return "Logistical issue: Injected volume does not equal Withdrawn volume"
    
    date_range = pd.date_range(injection_dates[0], withdrawal_dates[-1], freq='D')

    contract_value = 0
    stored_amount = 0
    injection_index = 0
    withdrawal_index = 0
    injection_dates = [datetime.strptime(d, "%Y-%m-%d") for d in injection_dates]
    withdrawal_dates = [datetime.strptime(d, "%Y-%m-%d") for d in withdrawal_dates]
    for i, date in enumerate(date_range):
        current_date = date
        if stored_amount > max_capacity:
            return "Logistical issue: Over max gas storage capacity"
        if current_date in injection_dates:
            stored_amount += inject_amounts[injection_index]
            contract_value -= injection_withdrawal_rate*inject_amounts[injection_index]
            contract_value -= inject_amounts[injection_index]*estimate_price(current_date.strftime("%Y-%m-%d"))
            injection_index += 1
        
        if current_date in withdrawal_dates:
            stored_amount -= withdrawal_amounts[withdrawal_index]
            contract_value -= injection_withdrawal_rate*withdrawal_amounts[withdrawal_index]
            contract_value += withdrawal_amounts[withdrawal_index]*estimate_price(current_date.strftime("%Y-%m-%d"))
            withdrawal_index += 1
        
        contract_value -= stored_amount*storage_cost_per_day

    return round(contract_value,2)

if __name__ == "__main__":
    print(price_storage_contract(['2024-09-01'],[300],['2025-01-01'],[300],0.1,500,0.001))
    #300 units gas bought at 11.51
    #Sold at 12.74
    #Stored for 122 days costing 36.6
    #Injection and withdrawal fees total 60
    #Should return a contract value of 272.4
