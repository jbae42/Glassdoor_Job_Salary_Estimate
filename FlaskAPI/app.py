import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle



def load_classifier():
    file_name = "models/classifier.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model1']
    return model

def load_sals_DS():
    file_name = "models/sal_DS.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model2']
    return model

def load_sals_SE():
    file_name = "models/sal_SE.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model3']
    return model

app = Flask(__name__)

@app.route('/predict', methods=['GET']) ##route would be the page on the website

def predict():
  # stub input features
    x = np.array(data_in).reshape(1,-1)
    # load model
    job_cls = load_classifier()
    sals_DS = load_sals_DS()
    sals_SE = load_sals_SE()
    res = dict()
    prediction = job_cls.predict(x)[0]
    if prediction == 1:
        res['Job Suggestion:'] = 'Data Scientist'
        sals = sals_DS.predict(x) * 1000
    else:
        res['Job Suggestion:'] = 'Software Engineer'
        sals = sals_SE.predict(x) * 1000
    res['Salary Estimate:'] = str(sals)
    response = json.dumps(res)
    return response, 200

#print(predict())

if __name__ == '__main__':
    application.run(debug=True)