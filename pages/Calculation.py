# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 19:31:07 2022

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

#df = pd.read_csv('bitcoin_data.csv')
df = yf.download('BTC-USD')

st.title("""
         #80% Training + 20% Testing
         """)

tab1, tab2, tab3 = st.tabs(["📈 Chart", "🗃 Data", "📈 Comparisson"])
    
tab1.subheader("Bitcoin Chart Price")
#last_rows = int(len(df)-1)
#tab1.chart = st.line_chart(df[0:last_rows].Date)

# Train Test Split

to_row = int(len(df)*0.9)

training_data = list(df[0:to_row]['Adj Close'])
testing_data = list(df[to_row:]['Adj Close'])

# split data into train and training data

fig = plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Price')
plt.plot(df[0:to_row]['Adj Close'], 'green', label='Train data')
plt.plot(df[to_row:]['Adj Close'], 'blue', label='Test data')
plt.legend()
tab1.pyplot(fig)

tab2.subheader("Bitcoin Data")
#tab2.write(df.describe())  
#with tab2.expander("See details"):
#    st.write(df)
    
model_prediction = []
n_test_obser = len(testing_data)

for i in range(n_test_obser):
  model = ARIMA(training_data, order = (4,1,0))
  model_fit = model.fit()
  output = model_fit.forecast()
  #first prediction
  yhat = list(output[0])[0]
  model_prediction.append(yhat)
  actual_test_value = testing_data[i]
  training_data.append(actual_test_value)

tab2.write(model_fit.summary())

fig = plt.figure(figsize=(20,12))
plt.grid(True)

date_range = df[to_row:].index

plt.plot(date_range, model_prediction, color = 'blue', marker = 0, linestyle='dashed', label='BTC Predicted Price')
plt.plot(date_range, testing_data, color = 'red', label='BTC Actual Price')

tab3.subheader("Predicted Price vs Actual Price")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
#plt.show()
tab3.pyplot(fig)

#report performance 
mape = np.mean(np.abs(np.array(model_prediction) - np.array(testing_data)) / np.abs(testing_data))*100
tab3.info('MAPE : '+str(mape))  #Mean Absolute Percentage Error

#Around 3.8% MAPE implies the model is abput 9.62% accurate in predictiing the test set obsevation