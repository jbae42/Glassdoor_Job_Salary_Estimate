# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:43:59 2021

@author: Andrew
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor


cwd = os.getcwd()


filepath = "C:/Users/Andrew/Desktop/EdX/Self Projects/Classification/Data_entry/test"
os.chdir(filepath)

#### Salary Estimate Regression Models

### Step 1. Read the CSV file and factorize by job type
df = pd.read_csv("df_cleaned_new.csv")
factor = pd.factorize(df['Job_Code'])
df['Job_Code'] = factor[0]

### Step 2. Selecting meaningful dataset (columns)
df_sal =df[['avg_salary','Job_Code','python','masters', 'statistic', 'SQL', 'spark', 'AWS', 'Tableau', 
                  'Hadoop', 'C_lang', 'Java', 'app', 'debug', 'HTML', 'object']]

### Step 3. Define dependent and independent variables
X_sal = df_sal.drop('avg_salary',axis=1)
y_sal = df_sal[['avg_salary']].values

### Step 4. Split train, test sets
X_sal_train, X_sal_test, y_sal_train, y_sal_test = train_test_split(X_sal, y_sal, test_size=0.2, random_state=42)

### Step 5. Try different modeling techniques for salary estimate regressor
rf = RandomForestRegressor(n_estimators=80, criterion='mae', max_features='sqrt')
rf.fit(X_sal_train,y_sal_train)
rf.score(X_sal_train, np.ravel(y_sal_train))
rf.score(X_sal_test, np.ravel(y_sal_test))
mean_absolute_error(y_sal_test, rf.predict(X_sal_test))

lm = linear_model.LinearRegression()
lm.fit(X_sal_train,y_sal_train)
lm.score(X_sal_train, np.ravel(y_sal_train))
lm.score(X_sal_test, np.ravel(y_sal_test))
mean_absolute_error(y_sal_test, lm.predict(X_sal_test))

xgb = XGBRegressor(verbosity = 0)
xgb.fit(X_sal_train,y_sal_train)
xgb.score(X_sal_train, np.ravel(y_sal_train))
xgb.score(X_sal_test, np.ravel(y_sal_test))
mean_absolute_error(y_sal_test, xgb.predict(X_sal_test))

cv_score = cross_val_score(xgb, X_sal_train,y_sal_train, cv=10)

## Try tuning the hyperparameters in random forest regressor
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
max_features = ['auto','sqrt']
max_depth = [2,4]
min_samples_split = [2,5]
min_samples_leaf = [1,2]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}


rf_Grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3, verbose=2, n_jobs=4)
rf_Grid.fit(X_sal_train, y_sal_train)
rf_Grid.score(X_sal_train, y_sal_train)
rf_Grid.score(X_sal_test,y_sal_test)

### Step 6. Results
'''
The first approach, RandomForestRegressor with hyperparameter tuning gave the best results.
A possible explanation to the low performance of the regressor is the nature of the salary data.
Since the dataset was extracted from glassdoor website where the salary data are provided as a range,
many salary data may have been affected by the location (state), rather than the job skills described in
the job description. This may have caused weak correlation between the listed job skills and salary.
'''