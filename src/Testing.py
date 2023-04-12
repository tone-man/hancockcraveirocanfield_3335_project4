import pandas as pd # Used for data analysis and manipulation.
import numpy as np # Used for arrays, matrices, and high-level math functions. 
import matplotlib.pyplot as plt # Used for graphics and visuals.
import os # Helps with operating system
import tkinter as tk # Used for GUI

from Preprocessor import preproccess
from sklearn.model_selection import train_test_split # Used for splitting data
from sklearn.preprocessing import MinMaxScaler # Used for scaling data
from sklearn.preprocessing import StandardScaler # Used also for scaling data 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


from Model import SVRModel, LogRegModel, MLPModel

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

SVRModel = SVRModel(X_train, y_train)
LogRegModel = LogRegModel(X_train, y_train)
MLPModel = MLPModel(X_train, y_train)

### Confusion matrix display dode for reference
SVR_basic_cf_matrix = confusion_matrix(y_test, y_pred)
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

def initGUI(g):
    windoe = tk.Tk()
    length = 1200
    width = 700
    dimension = str(length) + 'x' + str(width)
    # Creating window color is not white so we can see the images
    # Creating a Frame to view the images
    windoe.geometry(dimension)  
    windoe.title('3335_Project4')
    windoe.configure(background = 'white')

    gobutton = tk.Button(windoe, text = 'Go', width = 10, font = my_font1, command = lambda: button_click(stepcounter, g, labels, windoe))

    gobutton.grid(row = 7, column = 6, padx = 20, pady = 35, sticky = (N, S, E, W))


    windoe.mainloop() 