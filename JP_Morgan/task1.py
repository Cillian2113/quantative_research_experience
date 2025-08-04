import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

#Read csv file into df
gas = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'], date_format='%m/%d/%y')

#Define end date of our extrapolation
last_date = gas["Dates"].iloc[-1]
future_end = last_date + pd.DateOffset(years=1)

#Set date column as index for df, add days inbetween month end in
gas = gas.set_index('Dates')
date_range = pd.date_range(gas.index[0], gas.index[-1] , freq='D')
daily_prices = gas.reindex(date_range)

#Interpolate to find daily estimate of gas price
daily_prices['Prices'] = daily_prices['Prices'].interpolate(method='linear')

#Linear regression model does not take dates so we use numbers to represent our days and fit a line
t = np.arange(len(gas)).reshape(-1, 1)
model = LinearRegression()
model.fit(t, gas['Prices'])
future_dates = pd.date_range(gas.index[0], gas.index[-1] + pd.DateOffset(years=1), freq='ME')
t_future = np.arange(len(future_dates)).reshape(-1, 1)

# Predict using regression model
predicted_prices = model.predict(t_future)

#Take into account seasonal trends (price up in winter down in summer)
month_avg = gas.groupby(gas.index.month)['Prices'].mean()
seasonal_adjustment = np.array([month_avg[date.month] - month_avg.mean() for date in future_dates])
predicted_prices += seasonal_adjustment

#Create data frame for future predicted prices
gas_forecast = pd.DataFrame({
    'Prices': predicted_prices},
     index=future_dates)
gas_forecast.loc['2024-09-30', 'Prices'] = 11.8

#Again interpolate the monthly price forecasts so daily estimates can be made
forecast_date_range = pd.date_range(gas.index[0], future_end, freq='D')
daily_gas_forecast = gas_forecast.reindex(forecast_date_range)
daily_gas_forecast['Prices'] = daily_gas_forecast['Prices'].interpolate(method='linear')



def estimate_price(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    if date > last_date:
        return round(daily_gas_forecast.loc[date].iloc[0], 2)
    else:
        return round(daily_prices.loc[date].iloc[0], 2)


if __name__ == "__main__":
    input_date = input("Enter Enter date as YYYY-MM-DD")

"""
plt.figure(figsize=(10,5))
plt.plot(daily_prices.index, daily_prices["Prices"], marker='o', linestyle='-', alpha = 0.8, label = "Prices")
plt.plot(gas_forecast.index, gas_forecast['Prices'], marker='o', linestyle='-', alpha = 0.5, label = "Model")
plt.grid(True, linestyle='--', alpha=0.5) 
plt.xlabel("date")
plt.ylabel("price")
plt.savefig("plot.png")
"""