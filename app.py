# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:12:11 2022

@author: Venu
"""

import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd



model = pickle.load(open('model.pkl','rb'))
print('model is loaded')
app = Flask(__name__)
dataset = pd.read_csv('student-por-new.csv')

@app.route('/',methods=['GET'])
def index():
    school=str(request.args['school'])
    sex=str(request.args['sex'])
    age=int(request.args['age'])
    address=str(request.args['address'])
    famsize=str(request.args['famsize'])
    Pstatus=str(request.args['Pstatus'])
    Medu=int(request.args['Medu'])
    Fedu=int(request.args['Fedu'])
    Mjob=str(request.args['Mjob'])
    Fjob=str(request.args['Fjob'])
    reason=str(request.args['reason'])
    guardian=str(request.args['guardian'])
    traveltime=int(request.args['traveltime'])
    studytime=int(request.args['studytime'])
    failures=int(request.args['failures'])
    schoolsup=str(request.args['schoolsup'])
    famsup=str(request.args['famsup'])
    paid=str(request.args['paid'])
    activities=str(request.args['activities'])
    nursery=str(request.args['nursery'])
    higher=str(request.args['higher'])
    internet=str(request.args['internet'])
    famrel=int(request.args['famrel'])
    freetime=int(request.args['freetime'])
    goout=int(request.args['goout'])
    health=int(request.args['health'])
    absences=int(request.args['absences'])
    G1=int(request.args['G1'])
    G2=int(request.args['G2'])
    
    X = dataset.iloc[:, 0:-1].values
    ct1=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
    X=np.array(ct1.fit_transform(X))
    ct2=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
    X=np.array(ct2.fit_transform(X))
    ct3=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[5])],remainder='passthrough')
    X=np.array(ct3.fit_transform(X))
    ct4=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[7])],remainder='passthrough')
    X=np.array(ct4.fit_transform(X))
    ct5=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[9])],remainder='passthrough')
    X=np.array(ct5.fit_transform(X))
    ct6=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[13])],remainder='passthrough')
    X=np.array(ct6.fit_transform(X))
    ct7=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[18])],remainder='passthrough')
    X=np.array(ct7.fit_transform(X))
    ct8=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[23])],remainder='passthrough')
    X=np.array(ct8.fit_transform(X))
    ct9=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[27])],remainder='passthrough')
    X=np.array(ct9.fit_transform(X))
    ct10=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-14])],remainder='passthrough')
    X=np.array(ct10.fit_transform(X))
    ct11=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-13])],remainder='passthrough')
    X=np.array(ct11.fit_transform(X))
    ct12=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-12])],remainder='passthrough')
    X=np.array(ct12.fit_transform(X))
    ct13=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-11])],remainder='passthrough')
    X=np.array(ct13.fit_transform(X))
    ct14=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-10])],remainder='passthrough')
    X=np.array(ct14.fit_transform(X))
    ct15=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-9])],remainder='passthrough')
    X=np.array(ct15.fit_transform(X))
    ct16=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-8])],remainder='passthrough')
    X=np.array(ct16.fit_transform(X))
    test=[[school,
       sex,
       age,
       address,
       famsize,
       Pstatus,
       Medu,
       Fedu,
       Mjob,
       Fjob,
       reason,
       guardian,
       traveltime,
       studytime,
       failures,
       schoolsup,
       famsup,
       paid,
       activities,
       nursery,
       higher,
       internet,
       famrel,
       freetime,
       goout,
       health,
       absences,
       G1,
       G2]]    
    test=ct1.transform(test)
    test=ct2.transform(test)
    test=ct3.transform(test)
    test=ct4.transform(test)
    test=ct5.transform(test)
    test=ct6.transform(test)
    test=ct7.transform(test)
    test=ct8.transform(test)
    test=ct9.transform(test)
    test=ct10.transform(test)
    test=ct11.transform(test)
    test=ct12.transform(test)
    test=ct13.transform(test)
    test=ct14.transform(test)
    test=ct15.transform(test)
    test=ct16.transform(test)
    pred=model.predict(test)
    return jsonify(prediction=str(round(pred[0],2)))

if __name__ == "__main__":
    app.run(debug=True)