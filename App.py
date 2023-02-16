#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt
from plotly import graph_objs as go
import numpy as np 
from datetime import datetime
import pickle
from pickle import dump
from pickle import load
import plotly.express as px
from PIL import Image


# In[ ]:





# In[4]:


st.title('Apple_Stock_Forecasting Using Streamlit')
st.subheader('Apple Dataset')


# In[5]:


df = pd.read_csv('AAPL.csv')
df["Date"]=pd.to_datetime(df["Date"])
df.set_index('Date',inplace=True)


# In[9]:


image = Image.open("stockpic.jpg")
st.image(image, caption='Stock price prediction')
st.write("""In finance, accurately predicting stock prices is crucial for making informed investment decisions. Time series forecasting is a method used to predict future values based on past data and trends. This approach can be especially useful in analyzing the stock price of a publicly traded company like Apple Inc.""")


# In[10]:


st.sidebar.write('_Stock price prediction for upcoming 30 days_')


# In[11]:


company = st.sidebar.selectbox('Select the Company',('Apple', 'Tesla'))
type_data=st.sidebar.radio("Select type of the data",("Original","Predicted") )
graph=st.sidebar.radio("Select Visualization Type",("Graphical","Tabular") )
periods_input = st.sidebar.slider('How many days forecast do you want?',min_value = 1, max_value = 30)


# In[12]:


results = st.sidebar.button('show')


# In[13]:


st.sidebar.title("About :")
st.sidebar.subheader("Guided by: Neha Gupta")
st.sidebar.title("P-185 Team-2 :")
st.sidebar.subheader("Shakir, Vaishnavi, Vijay") 


# In[14]:



loaded_model = pickle.load(open('model_trained.pkl', 'rb'))
predict = loaded_model.forecast(periods_input)
pdf=pd.DataFrame(predict.values,index=pd.date_range('2019-12-31',  periods=periods_input), columns=["Close"])


# In[15]:


if (type_data=="Original"): 
      
    if (graph=="Graphical"): 
        
        if results:
            
            st.title("RESULTS")
    
            df1 = px.data.gapminder()
            fig = px.line(data,x=data.index,y=["Close","Open","High","Low"])
        
            tab1, tab2 = st.tabs(["plotly chart", "Line Chart"])
            
            with tab1:
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            with tab2:
                st.line_chart(data)                
            
    else:
        
        if results:
            
            st.title("RESULTS")
            st.write("Apple stock prices from 2012 to 2019")
            st.dataframe(data) 

            
else:
    
    if (graph=="Graphical"): 
    
        if results:
            
            st.title("RESULTS")
            df1 = px.data.gapminder()
            pp=pd.DataFrame(predict)
            
            fig1 = px.line(data, x=data.index , y=data["Close"],color_discrete_sequence=["blue"],labels="original" )
            fig2 = px.line(predict ,x= pd.date_range('2019-12-31',  periods=periods_input)  , y=predict.values,color_discrete_sequence=["red"], labels="Predicted" )
            fig3 = px.line(data[-100:], x=data[-100:].index , y=data["Close"][-100:],color_discrete_sequence=["blue"],labels="original" )

            fig = go.Figure(data = fig3.data + fig2.data)
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_layout(                      
                                title="AAPL",
                                xaxis_title="Date range",
                                yaxis_title="Close",
                                legend_title="Legend",
                                height=700
                              )
            
            fig4 = go.Figure(data = fig1.data + fig2.data)
            fig4.update_xaxes(rangeslider_visible=True)
            fig4.update_layout(                      
                                title="AAPL",
                                xaxis_title="Date range",
                                yaxis_title="Close",
                                legend_title="Legend",
                                height=700
                               )
                    
            tab1, tab2 = st.tabs(["Predicted data with past 100 days data", "Predicted data with Whole data"])
           
            with tab1:
              
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)
               

            with tab2:

                st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
          
    else:
        if results:
            st.title("RESULTS")
            st.dataframe(pdf)


# In[ ]:




