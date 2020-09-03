# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 23:35:20 2020

@author: Rajeeb
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Ask user for the image and open it

filename = input('Enter image path:')

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global x_val, y_val, b, g, r, clicked
        x_val = x
        y_val = y
        b,g,r = img[y,x]
        getColorName(r,g,b)
        
def getColorName(r,g,b):
    features = df[['R','G', 'B']].values
    values = df['color_name'].values
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(features, values)
    prediction = model.predict([[r,g,b]])
    print(prediction[0], r, g, b)

#Open the file for training
df = pd.read_csv('colors.csv', header = None)
df.columns = ['color', 'color_name', 'hex', 'R', 'G', 'B']
#print(df.head())

try:
    #Read image name
    img = cv2.imread(filename) 
    cv2.namedWindow('image')
    
    #Callback function when a mouse event happens
    cv2.setMouseCallback('image',draw_function)
    cv2.imshow('image', img)

    cv2.waitKey(20) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
except:
    print("No such file")
    cv2.destroyAllWindows()

