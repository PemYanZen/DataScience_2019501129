#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:30:27 2020

@author: pemayangdon
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr
from sklearn.decomposition import PCA



#Read train and test datasets
path = "/Users/pemayangdon/DataScience_2019501129/Introduction to Machine Learning/CodeCamp/Exercise1/house-prices-advanced-regression-techniques/";


trainx_df = pd.read_csv(path+"train.csv", index_col='Id');
trainy_df = trainx_df['SalePrice'];
trainx_df.drop('SalePrice', axis=1,inplace=True)
testx_df = pd.read_csv(path+"test.csv",index_col='Id')



#Preprocessing steps
#Step1: remove columns with null value ratio greater than provided limit
sample_size = len(trainx_df)

columns_with_null_values = [[col, float(trainx_df[col].isnull().sum())/ float(sample_size)] 
                            for col in trainx_df.columns 
                            if trainx_df[col].isnull().sum()]

columns_to_drop = [x for (x,y) in columns_with_null_values if y > .3]
trainx_df.drop(columns_to_drop,axis=1,inplace=True)
testx_df.drop(columns_to_drop,axis=1,inplace=True)

#step2:find all categorical columns and one hot encode them. 
#Before one hot encode fill all null values with dummy in those columns.  
#Some categorical columns in trainx_df may not have null values in trainx_df
# but have null values in testx_df. To overcome this problem we will add a 
#row to the trainx_df with all dummy values for categorical values.
# Once one hot encoding is complete drop the added dummy column

categorical_columns=[col for col in trainx_df.columns if 
                      trainx_df[col].dtype==object]

# #categorical_columns.append('MSSubClass')
ordinal_columns=[col for col in trainx_df.columns if col not in 
                  categorical_columns]

dummy_row = list()
for col in trainx_df.columns:
      if col in categorical_columns:
          dummy_row.append("dummy")
      else:
          dummy_row.append("")
          
new_row=pd.DataFrame([dummy_row], columns=trainx_df.columns)
trainx_df=pd.concat([trainx_df,new_row], axis=0, ignore_index=True)
testx_df=pd.concat([testx_df], axis=0, ignore_index=True)

for col in categorical_columns:
    trainx_df[col].fillna(value="dummy",inplace=True)
    testx_df[col].fillna(value="dummy",inplace=True)


#oneHotEncoding
enc = OneHotEncoder(drop='first', sparse=False)
enc.fit(trainx_df[categorical_columns])
# print(enc.get_feature_names(categorical_columns))
trainx_enc = pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
testx_enc = pd.DataFrame(enc.transform(testx_df[categorical_columns]))

trainx_enc.columns=enc.get_feature_names(categorical_columns)
testx_enc.columns=enc.get_feature_names(categorical_columns)

trainx_df = pd.concat([trainx_df[ordinal_columns],trainx_enc], axis=1,ignore_index=True)
testx_df = pd.concat([testx_df[ordinal_columns],testx_enc], axis=1,ignore_index=True)
trainx_df.drop(trainx_df.tail(1).index,inplace=True)
# #trainx_df.to_csv("encodedTrains.csv");


# As a third step of pre-processing fill all missing values for ordinal features
imputer = KNNImputer(n_neighbors=2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df_filled=pd.DataFrame(trainx_df_filled,columns = trainx_df.columns)
testx_df_filled = imputer.transform(testx_df)
testx_df_filled=pd.DataFrame(testx_df_filled,columns = testx_df.columns)
testx_df_filled.reset_index(drop=True, inplace=True)


# print(trainx_df_filled.isnull().sum())
# print(testx_df_filled.isnull().sum())
scaler = preprocessing.StandardScaler().fit(trainx_df_filled)

trainx_df_filled = scaler.transform(trainx_df_filled)
testx_df_filled = scaler.transform(testx_df_filled)

X_train, X_test, y_train, y_test = train_test_split(trainx_df_filled, trainy_df.values.ravel(), test_size=0.3, random_state=42)










