# Random Forest Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

"""## Importing the dataset"""

dataset = pd.read_csv('student-por-new.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

"""## One hot encoding"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

print(X[0])

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

print(X[0])

"""##Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""## Training the Random Forest Regression model on the whole dataset"""

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

"""##Determining accuracy"""

y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

t=[['GP','F',16,'U','GT3','T',4,4,'health','other','home','mother',1,1,0,'no','yes','no','no','yes','yes','yes',4,4,4,2,6,17,17]]
t=ct1.transform(t)
t=ct2.transform(t)
t=ct3.transform(t)
t=ct4.transform(t)
t=ct5.transform(t)
t=ct6.transform(t)
t=ct7.transform(t)
t=ct8.transform(t)
t=ct9.transform(t)
t=ct10.transform(t)
t=ct11.transform(t)
t=ct12.transform(t)
t=ct13.transform(t)
t=ct14.transform(t)
t=ct15.transform(t)
t=ct16.transform(t)
regressor.predict(t)

"""##Converting model to pkl format """

import pickle
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
