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

SVRModel = SVRModel()
LogRegModel = LogRegModel()
MLPModel = MLPModel()

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