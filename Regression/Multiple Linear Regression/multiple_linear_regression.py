# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
import seaborn as sb
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv("FuelConsumptionCo2.csv")

print(df.describe())
print(df.head())

print(df.dtypes)

#histogram view
hist_view = df[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
hist_view.hist()
mplt.pyplot.title("Histogram view of data")
mplt.pyplot.show()

#splitting data in test and train using 80-20 rule.
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#Multiple linear regression
linear_reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
linear_reg.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

#Ordinary least square method
y_hat= linear_reg.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linear_reg.score(x, y))