import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle
import os

filepath = "C:/Users/Andrew/Desktop/Github/project/Glassdoor_Job_Salary_Estimate/FlaskAPI"
os.chdir(filepath)

def load_classifier():
    file_name = "models/classifier.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model1']
    return model

def load_regressor():
    file_name = "models/regressor.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model2']
    return model


app = Flask(__name__)

@app.route('/predict', methods=['GET']) ##route would be the page on the website

def predict():
  # stub input features
    x = np.array(data_in).reshape(1,-1)
    # load model
    job_cls = load_classifier()
    sals = load_regressor()
    res = dict()
    prediction = job_cls.predict(x)[0]
    x2 = np.array([prediction]+data_in).reshape(1,-1)
    if prediction == 1:
        res['Job Suggestion:'] = 'Data Scientist'
    elif prediction == 2:
        res['Job Suggestion:'] = 'Software Engineer'
    else:
        res['Job Suggestion:'] = 'Data Analyst'
    sal_pred = sals.predict(x2) * 1000
    res['Salary Estimate:'] = str(sal_pred)
    response = json.dumps(res)
    return response, 200

#print(predict())

if __name__ == '__main__':
    application.run(debug=True)


### Note
'''
To run this script, first connect to the flask_env by following the steps below.
1. open Anaconda Prompt
2. change directory to the FlaskAPI folder
3. type 'conda activate flask_env'
4. run 'py wsgi.py'
5. open another Anaconda Prompt
6. type 'curl -X GET http://127.0.0.1:5080/predict'
'''