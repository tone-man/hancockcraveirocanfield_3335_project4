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

# Calling the preproccess function to get all the preprocessing code and store it into a new dataframe variable
df = preproccess()

# Splits the data to the label that you predict on (y) and all othr columns to predict with (X)
X = df.loc[:, df.columns != 'RainTomorrowFlag']
y = df.iloc[:, 18]
y = np.where(y == 0, 0, 1).astype(int) # Needs to happen to change from a dataframe to numpy array used for Train test split

# Scales the all data to a range of 0 - 1
mms = MinMaxScaler()
mms.fit(X)
Xmm = mms.transform(X)

# Splits the data into 70% for the training and 30% for the testing sets. the .astype ensures that there are no continous numbers in the array.
X_train, X_test, y_train, y_test = train_test_split(Xmm, y, stratify = y, test_size = 0.25, random_state = 0)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Double checks to see if the split has the same number of rows and columns 
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

## Training the models (SVR LogReg, MLP NN) for classifying data.
def SVRModel():
# SVR MODEL
    SVR_basic = LinearSVR(C = 50, epsilon = .1, random_state = 0)
    trained_SVR = SVR_basic.fit(X_train, y_train)
    return trained_SVR
def LogRegModel():
# LOGISTIC REGRESSION MODEL
    LogReg = LogisticRegression(penalty = 'l2', C = 50, solver = 'liblinear',  random_state = 0, max_iter = 500)
    trained_LogReg = LogReg.fit(X_train, y_train)
    return trained_LogReg

def MLPModel():
# NN through sklearn using MLP classifier
    mlp_basic = MLPClassifier(hidden_layer_sizes = (30, 20, 15), learning_rate = 'adaptive', random_state = 0)
    trained_MLP = mlp_basic.fit(X_train, y_train)
    return trained_MLP

trained_SVR = SVRModel()
trained_LogReg = LogRegModel()
trained_MLP = MLPModel()

y_pred = trained_SVR.predict(X_test).astype(int)
for i in range(len(y_pred)):
    if y_pred[i] >= 1:
        y_pred[i] = 1    
    else:
        y_pred[i] = 0      

print("The SVR intercept is", '%.4f'%(trained_SVR.intercept_))
print("The SVR coefficents are", trained_SVR.coef_)
print("The SVR Accuracy of the model is", '%.4f'%(accuracy_score(y_test, y_pred)))

y_pred2 = trained_LogReg.predict(X_test).astype(int)
print("LogReg accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred2)))

y_pred3 = trained_MLP.predict(X_test)
print("MLP accuracy is ", '%.4f'%(accuracy_score(y_test, y_pred3)))


### Confusion matrix display dode for reference
"""
SVR_basic_cf_matrix = confusion_matrix(y_test, y_pred)
# dispSVR = ConfusionMatrixDisplay(SVR_basic_cf_matrix, )
SVRcmd = ConfusionMatrixDisplay(SVR_basic_cf_matrix, display_labels = ['Yes', 'No'])
SVRcmd.plot()
SVRcmd.ax_.set(xlabel = 'Predicted', ylabel = 'True', title = 'SVR will it rain tomorrow')


LogR_cf_matrix = confusion_matrix(y_test, y_pred2)
LogRegcmd = ConfusionMatrixDisplay(LogR_cf_matrix, display_labels = ['Yes', 'No'])
LogRegcmd.plot(cmap = 'inferno')
LogRegcmd.ax_.set(xlabel = 'Predicted', ylabel = 'True', title = 'LogReg will it rain tomorrow')


MLPcf_matrix = confusion_matrix(y_test, y_pred3)
MLPcmd = ConfusionMatrixDisplay(MLPcf_matrix)
MLPcmd.plot(cmap = 'ocean')
MLPcmd.ax_.set(xlabel = 'Predicted', ylabel = 'True', title = 'MLP will it rain tomorrow')
plt.show()
"""