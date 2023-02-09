#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import streamlit as st
from plotly import graph_objs as go
import pickle
from pickle import dump
from pickle import load


# In[9]:


st.title('Apple_Stock_Forecasting Using Streamlit')
st.subheader('Apple Dataset')


# In[10]:


df = pd.read_csv('AAPL.csv')
st.write(df)
df.set_index('Date',inplace=True)


# In[11]:



fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index,y=df.Close,name='Close'))
fig.layout.update(title_text='Line graph of Close price',xaxis_rangeslider_visible=True)
st.plotly_chart(fig)


# In[15]:


loaded_model = pickle.load(open('C:/Users/shaki/apple_stock/trained_model.sav','rb'))
days = st.slider('Days for Prediction',0,200)


# In[16]:


if days > 1:
    fct = pd.DataFrame(loaded_model.forecast(days))
    st.write(fct)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fct.index,y=fct.predicted_mean,name='Forecast'))
    fig2.layout.update(title_text='Forecast of Closing price for the next given number of days',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index,y=df.Close,name='original_data'))
    fig3.add_trace(go.Scatter(x=fct.index,y=fct.predicted_mean,name='Forecast'))
    fig3.layout.update(title_text='Displaying forecast along with the original data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)


# In[ ]:





# In[ ]:



