# ooking into locaion variable and changing poosibly chaning it making a variable for location

import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import os # Helps with operating system

from sklearn.model_selection import train_test_split # Used for splitting data
from sklearn.preprocessing import MinMaxScaler # Used for scaling data
from sklearn.preprocessing import StandardScaler # Used also for scaling data 
from Preprocessor import preproccess

"""# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('weatherAUS.csv', encoding='utf-8')

Location_list = df['Location'].tolist()

print(Location_list)

        """

df = preproccess()

df.shape()

X = df.iloc[:, 0:24]
y = df.iloc[:, 24]
y = np.where(y == 0, 0, 1) # Needs to happen to change from a dataframe to numpy array used for Train test split

mms = MinMaxScaler()
mms.fit(X)
Xmm = mms.transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xmm, y, stratify = y, test_size = 0.2, random_state = 0)







