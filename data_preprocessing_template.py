#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:36:51 2019

@author: abul_fazal_saif
"""

# Data preprocessing

#importing the libraries

import numpy as np  # numpy is a mathematical tool to include any type of mathematics in our code 
import matplotlib.pyplot as plt #used to plot charts 
import pandas as pd  #library to import and manage datasets

data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values # created the matrix for the independent variables
y = data_set.iloc[:,3] # stored the last column in y

# Taking care of missing data
""" from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan , strategy='mean' )
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) """

#Encoding categorial data
""" from sklearn.preprocessing import LabelEncoder , OneHotEncoder
#LabelLearn simply categorises the data in terms of no. whick may lead to problems in equations (eg . 1 , 2 , 3 ....)
#OneHotEncode marks the categories to 1 on happenning. 

labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categorical_features = [0] , categories = 'auto')
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)  """

#splitting the datasets into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

"""  #feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  """