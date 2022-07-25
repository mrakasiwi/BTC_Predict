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
from datetime import datetime
import time

df = yf.download('BTC-USD')
#df = pd.read_csv('bitcoin_data.csv')
st.title("""
         #Bitcoin Prediction
         **Visually** show data of Bitcoin Price""")

tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
    
tab1.subheader("Bitcoin Chart Price")
last_rows = int(len(df)-1)
col1,_ = tab1.columns([40, 1])
data = df[0:]['Adj Close']

col1.line_chart(data)

tab2.subheader("Bitcoin Data")
tab2.write(df.describe())  
with tab2.expander("See details"):
    st.write(df)
    

