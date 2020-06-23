# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:27:30 2020

@author: Rajat sharma
"""
# Importing the libaries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the data set
dataset = pd.read_csv('placement_data')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Removing the Nan Values
from sklearn.impute import SimpleImputer
missing_values = SimpleImputer(missing_values = np.nan,
                               strategy = 'constant')
Y = Y.reshape(-1, 1)
missing_values = missing_values.fit(Y)
Y= missing_values.transform(Y)

# Changing the Cathogorical data
from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 0] = LabelEncoder_X.fit_transform(X[:, 0])
X[:, 2] = LabelEncoder_X.fit_transform(X[:, 2])
X[:, 4] = LabelEncoder_X.fit_transform(X[:, 4])
X[:, 5] = LabelEncoder_X.fit_transform(X[:, 5])
X[:, 7] = LabelEncoder_X.fit_transform(X[:, 7])
X[:, 8] = LabelEncoder_X.fit_transform(X[:, 8])
X[:, 10] = LabelEncoder_X.fit_transform(X[:, 10])
X[:, 12] = LabelEncoder_X.fit_transform(X[:, 12])

# Processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [5])],
                       remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [8])],
                       remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 1:]

# Spliting the data value in train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling, (meaning to scale a value from -1 to 1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

Y_pred[:, 0][Y_pred[:, 0] < 100000] = 0
plt.plot(Y_test, color = 'blue', label = 'Original')
plt.plot(Y_pred, color = 'green', label = 'Prediction')
plt.title('Placement Without Backward Propogation')
plt.xlabel('Score')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Using the Backward Elimination model for finding the Significant model
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((215, 1)).astype(int), values= X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5, 6, 10, 11, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5, 6, 10, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 5, 10, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 5, 10, 12, 13, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 5, 10, 12, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 5, 12, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 5, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 5, 14, 15]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

from sklearn.model_selection import train_test_split
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, Y, test_size=0.2, random_state=0)

#Now plotting the data
from sklearn.linear_model import LinearRegression
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, y_opt_train)
#Predicting the testset values
y_opt_pred = regressor_opt.predict(X_opt_test)



print(regressor_opt.coef_)
print(regressor_opt.intercept_)

y_opt_pred[:, 0][y_opt_pred[:, 0] < 100000] = 0
plt.plot(Y_test, color = 'blue', label = 'Original')
plt.plot(y_opt_pred, color = 'green', label = 'Prediction')
plt.title('Placement')
plt.xlabel('Score')
plt.ylabel('Salary')
plt.legend()
plt.show()




