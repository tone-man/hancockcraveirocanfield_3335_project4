import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import os # Helps with operating system
import tkinter as tk # Used for GUI
import random

from Preprocessor import preproccess
from sklearn.model_selection import train_test_split # Used for splitting data
from sklearn.preprocessing import MinMaxScaler # Used for scaling data
from sklearn.preprocessing import StandardScaler # Used also for scaling data 
from sklearn.preprocessing import MaxAbsScaler # Used also for scaling data
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from Model import SVRModel, LogRegModel, MLPModel
#TODO Rename Testing to ModelVisualizer
#TODO Add Neural Net Data
#TODO Plot the results

def getNewTestData():
    '''
    Creates a new testing data set to use for the model. Randomly picks a
    scalar, fits the contents of the data, and randomly slices it.
    '''
    #Generating scaled data
    transformerArray = [MinMaxScaler(), StandardScaler(), MaxAbsScaler()]

    transformer = transformerArray[random.randint(0, len(transformerArray) - 1)]
    transformer.fit(X)
    scaled_data = transformer.transform(X)
    
    #Split the scaled data
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, stratify = y, test_size = 0.25, random_state = random.randint(0, 9999))
    y_train = y_train.astype(int)
    y_test = y_test.astype(int) 
    
    return X_test, y_test

def testModel(X_test, y_test):
    '''
    Tests the models against a given test set
    '''
    y_pred = SVRModel.predict(X_test).astype(int)
    for i in range(len(y_pred)):
        if y_pred[i] >= 1:
            y_pred[i] = 1    
        else:
            y_pred[i] = 0      
    print("SVR Accuracy: ", '%.4f'%(accuracy_score(y_test, y_pred)))

    y_pred2 = LogRegModel.predict(X_test).astype(int)
    print("LogReg Accuracy: ", '%.4f'%(accuracy_score(y_test, y_pred2)))

    y_pred3 = MLPModel.predict(X_test)
    print("MLP: ", '%.4f'%(accuracy_score(y_test, y_pred3)))
    
    return y_pred, y_pred2, y_pred3


# MAIN METHOD
#---------------------------------------
# Calling the preproccess function to get all the preprocessing code and store it into a new dataframe variable

df = preproccess() #dataFrame of csv data + additional data
columns = df.columns #columns for data

# Splits the data to the label that you predict on (y) and all other columns to predict with (X)
X = df.loc[:, df.columns != 'RainTomorrowFlag']
y = df.iloc[:, 18]
y = np.where(y == 0, 0, 1).astype(int) # Needs to happen to change from a dataframe to numpy array used for Train test split

# Scales the all data to a range of 0 - 1 for training
transformer = MinMaxScaler()
transformer.fit(X)
scaled_data = transformer.transform(X)

# Splits the data into 70% for the training and 30% for the testing sets. the .astype ensures that there are no continous numbers in the array.
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, stratify = y, test_size = 0.25, random_state = 0)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("\n       TRAINING MODELS")
print("------------------------------")

SVRModel = SVRModel(X_train, y_train)
print("SVR Model Trained")
LogRegModel = LogRegModel(X_train, y_train)
print("LogReg Model Trained")
MLPModel = MLPModel(X_train, y_train)
print("MLP Model Trained")

print("\n       TESTING MODELS")
print("------------------------------")

#Run multiple tests
y_test = None
X_test = None
y_pred = None
y_pred2 = None
y_pred3 = None
for i in range(10):
    print("\n\t       RUN " + str(i))
    print("------------------------------")

    X_test, y_test = getNewTestData()
    y_pred, y_pred2, y_pred3 = testModel(X_test, y_test)
    
f, axes = plt.subplots(1, 3, figsize = (13, 5)) # Used to put the plots/confusion matrix on the same row

# Confusion Maxtrix for Basic AdalineSGD classifier
cf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cf_matrix, display_labels = ['Yes', 'No'])
disp.plot(ax = axes[0], xticks_rotation = 45, cmap = 'Blues')
disp.ax_.set_title('SVR will it rain tomorrow')
disp.ax_.set_xlabel('Predicted')
disp.ax_.set_ylabel('True')

# Confusion Matrix for Basic LogisticRegressionSGD classifier
cf_matrix2 = confusion_matrix(y_test, y_pred2)
disp2 = ConfusionMatrixDisplay(cf_matrix2, display_labels = ['Yes', 'No'])
disp2.plot(ax = axes[1], xticks_rotation = 45, cmap = 'Accent')
disp2.ax_.set_title('LogReg will it rain tomorrow')
disp2.ax_.set_xlabel('Predicted')
disp2.ax_.set_ylabel('True')

# Confusion Matrix for Basic LogisticRegressionSGD classifier
cf_matrix2 = confusion_matrix(y_test, y_pred3)
disp2 = ConfusionMatrixDisplay(cf_matrix2, display_labels = ['Yes', 'No'])
disp2.plot(ax = axes[2], xticks_rotation = 45, cmap = 'Reds')
disp2.ax_.set_title('MLP will it rain tomorrow')
disp2.ax_.set_xlabel('Predicted')
disp2.ax_.set_ylabel('True')
plt.subplots_adjust(wspace = 0.30, hspace = 0.5)

plt.plot()