import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS,cross_origin
import pickle

# load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
# app
app = Flask(__name__)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")
# routes
@app.route('/pred',methods=['POST','GET']) # route to show the review comments in a web UI
@cross_origin()
def main():
   if request.method == 'GET':
        return (render_template('index.html'))
   if request.method == 'POST':
        try:
            rate_marriage = request.form['rate_marriage']
            children = request.form['children']
            religious = request.form['religious']
            education = request.form['education']
            occupation = request.form['occupation']
            husband = request.form['occupation_husband']
            input_variables = pd.DataFrame([pd.Series([rate_marriage,children,religious,education,occupation,husband])])
            input_variables1 = pd.DataFrame(scaler.transform(input_variables))
            prediction = model.predict(input_variables1)
            return render_template('results.html', prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ', e)
            # return 'something is wrong'
            return e
   else:
        return render_template('index.html')

if __name__ == '__main__':

            # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app
