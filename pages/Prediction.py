# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:38:22 2022

@author: RAKA
"""
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mpld3
import streamlit.components.v1 as components
import pickle
from datetime import date, timedelta, datetime
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years

#st.markdown("# Page 2 ❄️")
#st.sidebar.markdown("# Page 2 ❄️")

st.title("""#Predict Bitcoin Price Up To Next Week""")


cols1,_ = st.sidebar.columns((20,1))
format = 'MMM DD, YYYY'
start_date = dt.datetime.now().date()  #  I need some range in the past
end_date = dt.datetime.now().date()+relativedelta(days=7)
max_days = end_date-start_date
predict_time = cols1.slider('Select date', min_value=start_date, value=end_date ,max_value=end_date, format=format)

st.sidebar.write("Predict price to: ", predict_time)

#df = pd.read_csv('bitcoin_data.csv')
df = yf.download('BTC-USD')

to_row = int(len(df)*0.9)
end_row = int(len(df))

training_data = list(df[:end_row]['Adj Close'])
testing_data = list(df[end_row-7:end_row+9]['Adj Close'])

training_date = list(df[to_row:].index)
testing_date = list(df[end_row-2:].index)

def date_range_list(start_date, end_date):
    # Return list of datetime.date objects between start_date and end_date (inclusive).
    curr_date = start_date
    while curr_date <= end_date:
        testing_date.append(curr_date)
        curr_date += timedelta(days=1)
    return testing_date

end_date = date_range_list(pd.to_datetime(testing_date[1]), predict_time)

for i in range(len(end_date)):
    # Insert each number at the start of list
    testing_data.insert(0, i)

model_prediction = []
n_test_obser = len(end_date)

for i in range(n_test_obser):
  model = ARIMA(training_data, order = (4,1,0))
  model_fit = model.fit()
  output = model_fit.forecast()
  #first prediction
  yhat = list(output[0])[0]
  model_prediction.append(yhat)
  actual_test_value = testing_data[i]
  training_data.append(actual_test_value)


fig = plt.figure(figsize=(20,12))
plt.grid(True)

date_range = end_date

#plt.plot(date_range, model_prediction, color = 'blue', marker = 0, linestyle='dashed', label='BTC Predicted Price')
#plt.plot(date_range, testing_data, color = 'red', label='BTC Actual Price')

plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
#plt.show()
#st.pyplot(fig)

#report performance 
#mape = np.mean(np.abs(np.array(model_prediction) - np.array(testing_data)) / np.abs(testing_data))*100
#st.info('MAPE : '+str(mape))  #Mean Absolute Percentage Error

#Around 3.8% MAPE implies the model is abput 9.62% accurate in predictiing the test set obsevation

col1, col2 = st.columns([3, 2])

data = pd.DataFrame({
  'date': np.array(date_range),
  'price': np.array(model_prediction)
})

data = data.rename(columns={'date':'index'}).set_index('index')


col1.subheader("Chart Prediction")
col1.line_chart(data)

col2.subheader("Price Prediction")
col2.write(data)