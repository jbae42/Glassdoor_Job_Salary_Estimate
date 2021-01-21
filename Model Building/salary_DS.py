# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:00:37 2021

@author: Andrew
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:18:09 2021

@author: Andrew
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#path = 'C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/Exploratory Data Anlysis/'
df = pd.read_csv("df_cleaned.csv")
df_DS = df[df['Job_Code'] == "DS"]


#choose relevant columns
print(df.columns)

df_model =df_DS[['avg_salary','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau', 'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

#get dummy data
df_dum = pd.get_dummies(df_model)


#train test split
X = df_dum.drop('avg_salary', axis = 1)
y = df_dum[['avg_salary']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#multiple linear regression
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X)
model.fit().summary()

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3))

#lasso regression
lm_l = Lasso(alpha=1.6)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3))

alpha = []
error =[]

for i in range(1,100):
    alpha.append(i/10)
    lml = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3)))

plt.plot(alpha,error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)] #best alpha at 1.6

#random forest
rf = RandomForestRegressor(n_estimators=80, criterion='mae', max_features='sqrt')
rf.fit(X,np.ravel(y))

np.mean(cross_val_score(rf, X_train, y_train, scoring ='neg_mean_absolute_error', cv=3))

#tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error',cv=3)

gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

#test ensembles
tpred_lm = lm.predict(X_test)
#tpred_lml = lml.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_rf)


X_test.columns
#['python', 'masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau','Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']
X_test.head(1)


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

df_ex = pd.DataFrame(data=example, index=[1])

print(rf.predict(df_ex)*1000)
