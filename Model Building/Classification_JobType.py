# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 21:37:37 2021

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

#### Classifier

### Step 1. Read the CSV file
df = pd.read_csv("df_cleaned_new.csv")

### Step 2. Categorize the job code
factor = pd.factorize(df['Job_Code'])
df['Job_Code'] = factor[0]
definitions = factor[1]

### Step 3. Split dependent variable and independent variables
X = df.iloc[:,14:].values
y = df.iloc[:,6].values

### Step 4. Split train, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 21)

### Step 5. Try out different classification approaches

## a. Random Forest Classifier - Improved with GridsearchCV
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


rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
rf_model.score(X_train,y_train)
rf_model.score(X_test,y_test)


rf_Grid = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv=3, verbose=2, n_jobs=4)
rf_Grid.fit(X_train, y_train)
rf_Grid.score(X_train,y_train)
rf_Grid.score(X_test,y_test)

rf_model = RandomForestClassifier(n_estimators=80, bootstrap=False, max_depth=4, max_features='sqrt', min_samples_leaf=2, min_samples_split=5)
rf_model.fit(X_train,y_train)
pred_gr = rf_model.predict(X_test)

rf_model.score(X_train,y_train)
rf_model.score(X_test,y_test)

mean_absolute_error(y_test, pred_gr)
mean_absolute_error(y_train, rf_model.predict(X_train))

## Best Case
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
classifier.score(X_train,y_train)
classifier.score(X_test,y_test)
mean_absolute_error(y_test, y_pred)

confusion_matrix(classifier.predict(X_test), np.ravel(y_test))

## b. K Nearest Neighbor Classification
knn1 = KNeighborsClassifier(n_neighbors = 1)
knn1.fit(X_train,y_train)
pred1 = knn1.predict(X_test)

mean_absolute_error(y_test, pred1)

confusion_matrix(knn1.predict(X_test), np.ravel(y_test))

classification_report(y_test, pred1)



error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_rate, color = 'black', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title("Error Rate vs. Number of K-Neighbors")
plt.xlabel("K-Neighbors")
plt.ylabel('Error Rate')

knn2 = KNeighborsClassifier(n_neighbors = 3)
knn2.fit(X_train,y_train)
pred2 = knn1.predict(X_test)

mean_absolute_error(y_test, pred2)

confusion_matrix(knn2.predict(X_test), np.ravel(y_test))

classification_report(y_test, pred2)


### Final Step: Pickel the Best Model for Productionization
import pickle
pickl = {'model1': classifier}
pickle.dump( pickl, open( 'classifier' + ".p", "wb" ) )
file_name = "classifier.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    job_cls = data['model1']