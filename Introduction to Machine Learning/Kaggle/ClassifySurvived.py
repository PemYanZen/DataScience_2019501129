#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:23:29 2020

@author: pemayangdon
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


import seaborn as sns

from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

#Read DataSets
def readDataSets(train_path, test_path,predict_col,index_col=None):
    if index_col==None:
        trainx_df=pd.read_csv(train_path)
        # trainy_df=trainx_df[predict_col]
        # trainy_df.hist()
        # trainx_df.drop(predict_col,axis=1,inplace=True)
        testx_df=pd.read_csv(test_path)
    else:
        trainx_df=pd.read_csv(train_path,index_col='PassengerId')
        # trainy_df=trainx_df[predict_col]
        # trainx_df.drop(predict_col,axis=1,inplace=True)
        testx_df=pd.read_csv(test_path,index_col='PassengerId')
    return trainx_df,testx_df

trainx_df,testx_df=readDataSets("train.csv","test.csv", predict_col='Survived',index_col="PassengerId")


#Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. 
#One way to do this is by filling in the mean age of all the passengers (imputation). 
#However we can be smarter about this and check the average age by passenger class. 
#For example:
    
    
# print(trainx_df.isnull().sum())

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=trainx_df,palette='winter')

#use average age values to impute based on Pclass for Age
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    

trainx_df['Age'] = trainx_df[['Age', 'Pclass']].apply(impute_age, axis=1)

testx_df['Age'] = testx_df[['Age', 'Pclass']].apply(impute_age, axis=1)


# converting categorical features to dummy variables
# print(trainx_df.info())
# print(pd.get_dummies(trainx_df['Embarked'],drop_first=True).head(100))


sex = pd.get_dummies(trainx_df['Sex'],drop_first=True)
embark = pd.get_dummies(trainx_df['Embarked'],drop_first=True)
trainx_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

trainx_df = pd.concat([trainx_df,sex,embark],axis=1)


sex = pd.get_dummies(testx_df['Sex'],drop_first=True)
embark = pd.get_dummies(testx_df['Embarked'],drop_first=True)
testx_df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

testx_df = pd.concat([testx_df,sex,embark],axis=1)



trainx_df.drop('Cabin', axis=1, inplace=True)
testx_df.drop('Cabin', axis=1, inplace=True)


imputer = KNNImputer(n_neighbors=2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df=pd.DataFrame(trainx_df_filled,columns=trainx_df.columns)

imputer.fit(testx_df)

testx_df_filled = imputer.transform(testx_df)
testx_df=pd.DataFrame(testx_df_filled,columns=testx_df.columns)

# testx_df.dropna(inplace=True)
# print(testx_df.isnull().sum())




X_train, X_test, y_train, y_test = train_test_split(trainx_df.drop('Survived',axis=1), 
                                                    trainx_df['Survived'], test_size=0.30, 
                                                    random_state=101)


def predictTestx(Model, testx_df):
    testpred=pd.DataFrame(Model.predict(testx_df)) 
    testpred = pd.DataFrame({'PassengerId': testpred.index, 'Survived': testpred[0]})
    testpred.to_csv("test_pred.csv",index=False)
    
    
# C=1.0
#Training and Prediction
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.001,
          verbose=0, warm_start=False)

model.fit(X_train,y_train)

predictTestx(model, testx_df)

predictions = model.predict(X_test)


accuracy=accuracy_score(y_test,predictions)
# print(accuracy)


#optimize hyper parameters of a Logistic Regression model using Grid Search 
std_slc = StandardScaler()
pca = decomposition.PCA()
logistic_Reg = linear_model.LogisticRegression()
pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('logistic_Reg', logistic_Reg)])
n_components = list(range(1,X_train.shape[1]+1,1))
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
    
parameters = dict(pca__n_components=n_components,
                      logistic_Reg__C=C,
                      logistic_Reg__penalty=penalty)

clf_GS = GridSearchCV(pipe, parameters)
# clf_GS.fit(X_train,y_train)


# print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
# print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
# print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
# print(); print(clf_GS.best_estimator_.get_params()['logistic_Reg'])






