#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Create the custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean = True, with_std = True):
        self.scaler = StandardScaler(copy,with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]
    
# Create the absenteeism class to do the modeling and predict the new data
class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        # read the saved model and scaler file
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    # Take a new csv file and preprocess it
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter = ',')
        # make a copy of the data
        self.df_with_predictions = df.copy()
        # drop the ID column
        df.drop(['ID'], axis = 1, inplace = True)
        # add a column with 'NaN' strings to preserve the code we've created in the previous section
        df['Absenteesim Time in Hours'] = 'NaN'
        
        # create a separate DF, containing dummy value for all available regions
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        
        # split reason_columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1) # implement max along the horizontal line
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        
        # drop the 'Reason for Absence' column from df to avoid multicollinearity
        df.drop(['Reason for Absence'], axis = 1, inplace = True)
        
        # concatenate df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1) # axis = 1 is to tell its to add as columns
        
        # assign names to the 4 reason type
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2',
                        'Reason_3', 'Reason_4']
        df.columns = column_names
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4',
                                  'Date', 'Transportation Expense', 'Distance to Work', 'Age',
                                  'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        # convert date into datetime
        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
        
        # create a list with month
        list_months = [df['Date'][i].month for i in range(df.shape[0])]
        
        # insert months in a new column in df
        df['Month Value'] = list_months
        
        # create a new feature called 'Day of the Week'
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        
        # Drop the date column
        df.drop(['Date'], axis = 1, inplace = True)
        
        # r-order the columns in df
        columns_name_update = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value',
                               'Day of the Week','Transportation Expense', 'Distance to Work',
                               'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[columns_name_update]
        
        # map education variables; the resunt in a dummy
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        #replace the NaN values
        df = df.fillna(value = 0)
        
        # drop the original absenteeism time        
        # drop the variables we decided we do not need
        df.drop(['Absenteeism Time in Hours', 'Day of the Week',
                 'Daily Work Load Average', 'Distance to Work'], axis = 1, inplace = True)
        
        # this is if you want to call the preprocessed data
        self.preprocessed_data = df.copy()
        
        # to use it in the next functions
        self.data = self.scaler.transform(df)
        
    # function to output the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred

    # a function outputing 0 or 1 based on the model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data 

