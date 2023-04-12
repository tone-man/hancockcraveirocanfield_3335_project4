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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

from Model import SVRModel, LogRegModel, MLPModel

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

# Double checks to see if the split has the same number of rows and columns 
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

SVRModel = SVRModel(X_train, y_train)
LogRegModel = LogRegModel(X_train, y_train)
#MLPModel = MLPModel(X_train, y_train)

y_pred = SVRModel.predict(X_test).astype(int)
for i in range(len(y_pred)):
    if y_pred[i] >= 1:
        y_pred[i] = 1    
    else:
        y_pred[i] = 0      

print("The SVR intercept is", '%.4f'%(SVRModel.intercept_))
print("The SVR coefficents are", SVRModel.coef_)
print("The SVR Accuracy of the model is", '%.4f'%(accuracy_score(y_test, y_pred)))

y_pred2 = LogRegModel.predict(X_test).astype(int)
print("LogReg accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred2)))

#y_pred3 = MLPModel.predict(X_test)
#print("MLP accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred3)))


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

for i in range(10):
    X_test, y_test = getNewTestData()
    testModel(X_test, y_test)
  
def getNewTestData():
    '''
    Creates a new testing data set to use for the model. Randomly picks a
    scalar, fits the contents of the data, and randomly slices it.
    '''
    #Generating scaled data
    transformerArray = [MinMaxScaler(), StandardScaler(), ABVScalar()]

    transformer = tranformerArray[random.getInt(0, len(tranformerArray) - 1)]
    transformer.fit(X)
    scaled_data = transformer.transfrom(X)
    
    #Split the scaled data
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, stratify = y, test_size = 0.25, random_state = random.randint(0, 9999))
    y_train = y_train.astype(int)
    y_test = y_test.astype(int) 
    
    return X_test, y_test

def testModels(X_test, y_test):
    '''
    Tests the models against a given test set
    '''
    y_pred = SVRModel.predict(X_test).astype(int)
    for i in range(len(y_pred)):
        if y_pred[i] >= 1:
            y_pred[i] = 1    
        else:
            y_pred[i] = 0      
    print("The SVR Accuracy of the model is", '%.4f'%(accuracy_score(y_test, y_pred)))

    y_pred2 = LogRegModel.predict(X_test).astype(int)
    print("LogReg accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred2)))

    y_pred3 = MLPModel.predict(X_test)
    print("MLP accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred3)))
