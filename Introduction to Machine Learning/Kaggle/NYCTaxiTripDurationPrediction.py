#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 5 20:58:25 2020

@author: pemayangdon
"""


import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr
from sklearn.linear_model import Ridge



#read datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head(5))

# print(train.info())


#data preprocessing steps

#check missing values
# print(train.isnull().sum())
# print(test.isnull().sum())


#encode categorical data
encode = LabelEncoder()
encode.fit(train['store_and_fwd_flag'])
train['store_and_fwd_flag'] = encode.transform(train['store_and_fwd_flag'])
test['store_and_fwd_flag'] = encode.transform(test['store_and_fwd_flag'])
# print(train.head())

train['trip_duration'] = np.log1p(train['trip_duration'].values) 


#Feature creation
# print(train.columns)
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

train['month'] = train['pickup_datetime'].dt.month
train['day'] = train['pickup_datetime'].dt.day
train['weekday'] = train['pickup_datetime'].dt.weekday
train['hour'] = train['pickup_datetime'].dt.hour
train['minute'] = train['pickup_datetime'].dt.minute

test['month'] = test['pickup_datetime'].dt.month
test['day'] = test['pickup_datetime'].dt.day
test['weekday'] = test['pickup_datetime'].dt.weekday
test['hour'] = test['pickup_datetime'].dt.hour
test['minute'] = test['pickup_datetime'].dt.minute

# print(train.head(4))


new_features_train = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month", "day", "weekday", "hour", "minute"]
new_features_test = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "month", "day", "weekday", "hour", "minute"]

trainx = train[new_features_train]
trainy = train["trip_duration"]
testx = test[new_features_test]


#Scale features through standard scores
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(trainx)
X_test_scaled = scaler.transform(testx)


# split the trainx_df into two parts to build a model and test how is it working to pick best model
def splitTrainAndTest(trainx_df, trainy_df,split_ratio=0.3):
    X_train, X_test, y_train, y_test = train_test_split(trainx_df, trainy_df.values.ravel(), test_size=split_ratio, random_state=42)
    return X_train, X_test, y_train, y_test


# Fit Linear Regression Model
def getLinearRegressionModel(X_train, y_train):
    reg=LinearRegression()
    reg = LinearRegression().fit(X_train, y_train)
    return reg



# Fit Regularized Linear regression using Ridge Regression
def getRidgeRegressionModel(X_train, y_train,tolerence=0.0001,reg_par=0.5):
    reg = Ridge(alpha=reg_par,tol=0.01)
    reg = reg.fit(X_train, y_train)
    return reg
    
# Tune regularization Parameter based on R^2 value or mse
def getRSquareandMSEVsAlphaPlots(X_train, X_test, y_train, y_test,alpha_start=0.1,alpha_end=10,jumps=10):
    score_train=[]
    score_test=[]
    mse_train=[]
    mse_test=[]
    alpha=[]
    for sigma in np.linspace(alpha_start, alpha_end, jumps):
        alpha.append(sigma)
        Ridge_model=getRidgeRegressionModel(X_train, y_train,reg_par=sigma)
        score_train.append(round(Ridge_model.score(X_train, y_train),10))
        score_test.append(round(Ridge_model.score(X_test, y_test),10))
        mse_train.append(round(mserr(y_train,Ridge_model.predict(X_train)),4))
        mse_test.append(round(mserr(y_test,Ridge_model.predict(X_test)),4))
        
    print(alpha,'\n', score_train, '\n',score_test,'\n', mse_train, '\n',mse_test) 
    plt.figure(1)
    plt.plot(alpha, score_train, 'g--',label="train_score")
    plt.plot(alpha, score_test, 'r-o',label="test_score")
    plt.xlabel='Alpha'
    plt.legend()
    plt.figure(2)
    plt.plot(alpha, mse_train, 'y--',label="train_mse")
    plt.plot(alpha, mse_test, 'c-o',label="test_mse")
    plt.xlabel='Alpha'
    plt.legend()
    plt.show()
    
# Predict values
def predictTestx(Model, testx_df):
    testpred=pd.DataFrame(Model.predict(testx_df)) 
    return testpred
    
    
X_train, X_test, y_train, y_test=splitTrainAndTest(trainx, trainy,split_ratio=0.3)

LRModel=getLinearRegressionModel(X_train, y_train)
# y_pred = predictTestx(LRModel, testx)

RidgeModel=getRidgeRegressionModel(X_train, y_train,tolerence=0.0001,reg_par=0.003)
y_pred = predictTestx(RidgeModel, testx)

# getRSquareandMSEVsAlphaPlots(X_train, X_test, y_train, y_test,alpha_start=0.001,alpha_end=0.003,jumps=10)


submission = pd.DataFrame({'id': test.id, 'trip_duration': np.expm1(y_pred[0])})
# print(submission.shape)
# submission.to_csv('submission.csv', index=False)


