# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:05:49 2021

@author: Andrew
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


#path = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/Exploratory Data Anlysis/'
df = pd.read_csv("df_cleaned.csv")

#choose relevant columns
print(df.columns)

df_model =df[['Job_Code','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau',
       'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

#get dummy data
df_dum = pd.get_dummies(df_model)


#train test split
X = df_dum.drop('Job_Code_DS', axis = 1).drop('Job_Code_SE', axis = 1)
y = df_dum[['Job_Code_DS']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)

## Logistic Regression
lr.fit(X_train, np.ravel(y_train))
lr.score(X_train, np.ravel(y_train))
lr.score(X_test, np.ravel(y_test))

preds = lr.predict(X_test)
pd.crosstab(preds, np.ravel(y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix(lr.predict(X_test), np.ravel(y_test))

## RandomForest Classification
rfc.fit(X_train,np.ravel(y_train))
rfc.score(X_train,np.ravel(y_train))
rfc.score(X_test,np.ravel(y_test))

confusion_matrix(rfc.predict(X_test), np.ravel(y_test))


## Linear SVC
svc.fit(X_train,np.ravel(y_train))
svc.score(X_train,np.ravel(y_train))
svc.score(X_test,np.ravel(y_test))

confusion_matrix(svc.predict(X_test), np.ravel(y_test))


example = {'python': 1,
           'masters': 1,
           'statistic': 1,
           'SQL': 1,
           'spark':0,
           'AWS':0,
           'Tableau':0,
           'Hadoop':0,
           'C_lang':0,
           'Java':0,
           'app':0,
           'debug':0,
           'HTML':0,
           'object':1}


example2 = {'python': 1,
           'masters': 0,
           'statistic': 0,
           'SQL': 1,
           'spark':0,
           'AWS':0,
           'Tableau':0,
           'Hadoop':0,
           'C_lang':1,
           'Java':1,
           'app':1,
           'debug':0,
           'HTML':0,
           'object':1}

df_ex = pd.DataFrame(data=example, index=[1])
df_ex2 = pd.DataFrame(data=example2, index=[1])

#print(rfc.predict(df_ex))
#print(rfc.predict(df_ex2))

data_input = df_ex2

prediction = rfc.predict(X_test[:1])[0]

print(X_test[:1])
print(prediction)
res = {'response': prediction}
print(res)

if rfc.predict(data_input) == 1:
    print('Data Scientist')
elif rfc.predict(data_input) == 0:
    print('Sofrware Engineer')