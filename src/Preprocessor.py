import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import seaborn as sns # Used for visuals as well as exploring the data.
import os # Helps with operating system

# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]

# For other columns with missing values, fill them in with column mean
df=df.fillna(df.mean())

# Create a flag for RainToday and RainTomorrow, note RainTomorrowFlag will be our target variable
df['RainTodayFlag']=df['RainToday'].apply(lambda x: 1 if x=='Yes' else 0)
df['RainTomorrowFlag']=df['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)

# Show a snaphsot of data
df.head(10)

#Drop Date, Location, Evaporation, and Sunshine Columns
df=df.drop(['Date','Location','Evaporation','Sunshine'], axis=1)
#Make new column by getting the average of MinTemp and Maxtemp
df['TempAvg'] = 0
for index, row in df.iterrows():
    row['TempAvg'] = (row['MinTemp'] + row['MaxTemp'])/2
#Make WindGustDir, WindDir9am, and WindDir3pm into Enums values 1 - 16
#N = 1, NNE = 2, NE = 3, ENE = 4, E = 5, ESE = 6, SE = 7, SSE = 8, S = 9, SSW = 10, SW = 11, WSW = 12, W = 13, WNW = 14, NW = 15, NNW = 16
df['WindGustDir'] = df['WindGustDir'].replace(['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])