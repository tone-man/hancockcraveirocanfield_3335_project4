# ooking into locaion variable and changing poosibly chaning it making a variable for location

import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import os # Helps with operating system

from Preprocessor import preproccess
from sklearn.model_selection import train_test_split # Used for splitting data
from sklearn.preprocessing import MinMaxScaler # Used for scaling data
from sklearn.preprocessing import StandardScaler # Used also for scaling data 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
df = preproccess()

df1 = df.loc[:, df.columns != 'RainTomorrowFlag']
columns = df1.columns 

## Training the models (SVR LogReg, MLP NN) for classifying data.
def SVRModel(X_train, y_train):
# SVR MODEL
    SVR_basic = LinearSVR(C = 50, epsilon = .1, random_state = 0)
    trained_SVR = SVR_basic.fit(X_train, y_train)
    numbers = SVR_basic.coef_

    print("The SVR intercept is", '%.4f'%(SVR_basic.intercept_))
    print("The SVR coefficents are ")
    for i in range(24):
        print(columns[i],":", '%.4f'%(numbers[i]))
    return trained_SVR

def LogRegModel(X_train, y_train):
# LOGISTIC REGRESSION MODEL
    LogReg = LogisticRegression(penalty = 'l2', C = 50, solver = 'liblinear',  random_state = 0, max_iter = 500)
    trained_LogReg = LogReg.fit(X_train, y_train)
    print("The LogReg probabilites are ", LogReg.get_params(deep=True))

    return trained_LogReg

def MLPModel(X_train, y_train):
# NN through sklearn using MLP classifier
    mlp_basic = MLPClassifier(hidden_layer_sizes = (30, 20, 15), learning_rate = 'adaptive', random_state = 0)
    trained_MLP = mlp_basic.fit(X_train, y_train)
    return trained_MLP


