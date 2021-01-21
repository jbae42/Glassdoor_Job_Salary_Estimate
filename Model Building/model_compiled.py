# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:48:58 2021

@author: Andrew
"""

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor


#Step 1. Read the CSV file
df = pd.read_csv("df_cleaned.csv")

#Step 2. Divide the dataset for salary regression model
df_DS = df[df['Job_Code'] == "DS"]
df_SE = df[df['Job_Code'] == "SE"]

#Step 3. Extract meaningful columns
df_class_model =df[['Job_Code','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau',
                    'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

df_DS_model =df_DS[['avg_salary','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau', 
                 'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

df_SE_model =df_SE[['avg_salary','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau', 
                 'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

#Step 4. Create dummy data
df_class_dum = pd.get_dummies(df_class_model)
df_DS_dum = pd.get_dummies(df_DS_model)
df_SE_dum = pd.get_dummies(df_SE_model)

#Step 5. Train-test set split
X_class = df_class_dum.drop('Job_Code_DS', axis = 1).drop('Job_Code_SE', axis = 1)
y_class = df_class_dum[['Job_Code_DS']].values
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

X_DS = df_DS_dum.drop('avg_salary', axis = 1)
y_DS = df_DS_dum[['avg_salary']].values
X_DS_train, X_DS_test, y_DS_train, y_DS_test = train_test_split(X_DS, y_DS, test_size=0.2, random_state=42)

X_SE = df_SE_dum.drop('avg_salary', axis = 1)
y_SE = df_SE_dum[['avg_salary']].values
X_SE_train, X_SE_test, y_SE_train, y_SE_test = train_test_split(X_SE, y_SE, test_size=0.2, random_state=42)

#Step 6. Build RandomForest Models
rfc_cls = RandomForestClassifier(n_estimators=100)
rfc_cls.fit(X_cls_train,np.ravel(y_cls_train))
rfc_cls.score(X_cls_train,np.ravel(y_cls_train))
rfc_cls.score(X_cls_test,np.ravel(y_cls_test))

#confusion_matrix(rfc_cls.predict(X_cls_test), np.ravel(y_cls_test))

rf_DS = RandomForestRegressor(n_estimators=80, criterion='mae', max_features='sqrt')
rf_DS.fit(X_DS_train,np.ravel(y_DS_train))
rf_DS.score(X_DS_train,np.ravel(y_DS_train))
rf_DS.score(X_DS_test,np.ravel(y_DS_test))

#confusion_matrix(rf_DS.predict(X_DS_test), np.ravel(y_DS_test))

rf_SE = RandomForestRegressor(n_estimators=80, criterion='mae', max_features='sqrt')
rf_SE.fit(X_SE_train,np.ravel(y_SE_train))
rf_SE.score(X_SE_train,np.ravel(y_SE_train))
rf_SE.score(X_SE_test,np.ravel(y_SE_test))

#confusion_matrix(rf_SE.predict(X_SE_test), np.ravel(y_SE_test))



import pickle
pickl = {'model1': rfc_cls}
pick2 = {'model2': rf_DS}
pick3 = {'model3': rf_SE}
pickle.dump( pickl, open( 'classifier' + ".p", "wb" ) )
pickle.dump( pick2, open( 'sal_DS' + ".p", "wb" ) )
pickle.dump( pick3, open( 'sal_SE' + ".p", "wb" ) )


file_name = "classifier.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    job_cls = data['model1']

file_name = "sal_DS.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    job_cls = data['model2']
    
file_name = "sal_SE.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    job_cls = data['model3']



### Get user input
# =============================================================================
# print("please answer each questino with Y or N")
# time.sleep(1)
# print("Do you have experience with python?")
# python = input().lower()
# print("Do you have a master's degree in the related field?")
# masters = input().lower()
# print("Do you have experience with developing statstical models (predictive, regression, etc.)?")
# statistic = input().lower()
# print("Do you have working knowledge of SQL?")
# SQL = input().lower()
# print("Do you have experience with spark?")
# spark = input().lower()
# print("Do you have experience with AWS?")
# AWS = input().lower()
# print("Do you have experience with Tableau?")
# Tableau = input().lower()
# print("Do you have experience with Hadoop?")
# Hadoop = input().lower()
# print("Do you have experience with C, C++, or C#?")
# C_lang = input().lower()
# print("Do you have experience with Java?")
# Java = input().lower()
# print("Do you have experience with application development?")
# app = input().lower()
# print("Do you have experience with debugging?")
# debug = input().lower()
# print("Do you have experience with HTML?")
# HTML = input().lower()
# print("Do you have expereince with object-oriented programming?")
# object_orient = input().lower()
# 
# in_data = [python, masters, statistic, SQL, spark, AWS, Tableau, Hadoop, C_lang, Java, app, debug, HTML, object_orient]
# in_data = [1 if ele == 'y' else 0 for ele in in_data]
# df_data = np.ravel(pd.DataFrame(in_data, index=['python', 'masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau','Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']))
# 
# 
# ## Run the models with input data
# cls_prediction = rfc_cls.predict(df_data.reshape(1,-1))
# 
# if cls_prediction == 1:   
#     sal_pred = rf_DS.predict(df_data.reshape(1,-1)) * 1000
#     print("With your qualifications, the model suggests a Data Scientist Job with a salary estimate of ${}".format(sal_pred))
# else:
#     sal_pred = rf_SE.predict(df_data.reshape(1,-1)) * 1000
#     print("With your qualifications, the model suggests a Software Engineer Job with a salary estimate of ${}".format(sal_pred))
# =============================================================================

















