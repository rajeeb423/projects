# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 23:35:20 2020

@author: Rajeeb
"""

import pandas as pd
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

# Ask user for the image and open it

filename = input('Enter image path:')

def draw_function(event, x, y, flags, param):
    cv2.imshow('image', img)
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global x_val, y_val, b, g, r, clicked
        x_val = x
        y_val = y
        b,g,r = [int(i) for i in img[y,x]]
        getColorName(r,g,b)
        
def getColorName(r,g,b):
    features = df[['R','G', 'B']].values
    values = df['color_name'].values
    
    #Train the model
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(features, values)
    
    #Do the prediction
    prediction = model.predict([[r,g,b]])
    
    #Display the answer
    text = prediction[0] + ' R='+ str(r) +' G='+ str(g) +' B='+ str(b)
    
    #rectangle(image, startpoints, endpoints, color, thickness)
    cv2.rectangle(img, (20, 20), (750, 60), (b,g,r), -1) # -1 means it will fill the entire rectangle

    cv2.putText(img, text, (50,50), color = ((255,255,255) if b+g+r < 600 else (0,0,0)), fontScale = 0.8, fontFace = 2)

#Open the file for training
df = pd.read_csv('colors.csv', header = None)
df.columns = ['color', 'color_name', 'hex', 'R', 'G', 'B']
#print(df.head())

try:
    #Read image name
    img = cv2.imread(filename) 
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    
    #Callback function when a mouse event happens
    cv2.setMouseCallback('image',draw_function)
    
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

    
except:
    print("No such file")
    cv2.destroyAllWindows()

