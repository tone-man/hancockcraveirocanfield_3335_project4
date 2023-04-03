import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import os # Helps with operating system

# Set Pandas options to display more columns
pd.options.display.max_columns = 50

# Read in the weather data csv
df = pd.read_csv('weatherAUS.csv', encoding = 'utf-8')

# Drop records where target RainTomorrow=NaN
df = df[pd.isnull(df['RainTomorrow'])==False]

# Replaces NaN with -1 to represent no input to help with over fitting the data on columns where NaN's are common
df['Cloud9am'] = df['Cloud9am'].fillna(-1)
df['Cloud3pm'] = df['Cloud3pm'].fillna(-1)

# For other columns with missing values, fill them in with column mean
df = df.fillna(df.mean())

# Create a flag for RainToday and RainTomorrow, note RainTomorrowFlag will be our target variable
df['RainTodayFlag'] = df['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0).astype(float)
df['RainTomorrowFlag'] = df['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0).astype(float)

# Drop Date, Location(maybe change this), Evaporation, and Sunshine Columns
# Also dropping RainToday, and RainTomorrow since we made numerical flag variables for them
df = df.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'RainToday', 'RainTomorrow'], axis = 1)

# Make WindGustDir, WindDir9am, and WindDir3pm into Enums values 1 - 16
# N = 1, NNE = 2, NE = 3, ENE = 4, 
# E = 5, ESE = 6, SE = 7, SSE = 8, 
# S = 9, SSW = 10, SW = 11, WSW = 12, 
# W = 13, WNW = 14, NW = 15, NNW = 16
df['WindGustDir'] = df['WindGustDir'].replace(['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'], 
                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
df['WindDir9am'] = df['WindDir9am'].replace(['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'], 
                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
df['WindDir3pm'] = df['WindDir3pm'].replace(['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'], 
                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

# Make new column by getting the average of MinTemp and Maxtemp
df['AvgTemp'] = df[['MinTemp', 'MaxTemp','Temp9am', 'Temp3pm']].mean(axis = 1)
df['AvgWindDir'] = df[['WindDir9am', 'WindDir3pm']].mean(axis = 1)
df['AvgWindSpeed'] = df[['WindSpeed9am', 'WindSpeed3pm']].mean(axis = 1)
df['AvgHumidity'] = df[['Humidity9am', 'Humidity3pm']].mean(axis = 1)
df['AvgPressure'] = df[['Pressure9am', 'Pressure3pm']].mean(axis = 1)
df['AvgCloud'] = df[['Cloud9am', 'Cloud3pm']].mean(axis = 1)


# Show a snaphsot of data
print(df.head(10))
print(df.dtypes)