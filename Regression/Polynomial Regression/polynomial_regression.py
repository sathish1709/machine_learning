# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumptionCo2.csv")

print(df.describe())
print(df.head())

print(df.dtypes)

#histogram view
#finding relationship between two columns in a dataframe
#x axis : predictive variable/independent variable
#y axis : target variable
x = df["ENGINESIZE"]
y = df["CO2EMISSIONS"]
mplt.pyplot.scatter(x,y)
mplt.pyplot.xlabel("engine size")
mplt.pyplot.ylabel("co2 emission")
mplt.pyplot.title("Relationship between engine size and CO2EMISSIONS: Scatter Plot")
pyplot.show()
#splitting data in test and train using 80-20 rule.
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(train[['ENGINESIZE']])
test_y = np.asanyarray(train[['CO2EMISSIONS']])

#if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2
poly_reg = PolynomialFeatures(degree=2)

#**fit_transform** takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).
train_x_poly = poly_reg.fit_transform(train_x)

#Now follow simple linear regression process
linear_reg = linear_model.LinearRegression()
linear_reg.fit (train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

mplt.pyplot.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
X = np.arange(0.0, 10.0, 0.1)
Y = linear_reg.intercept_[0]+ linear_reg.coef_[0][1]*X+ linear_reg.coef_[0][2]*np.power(X, 2) #Polynomial equation
mplt.pyplot.plot(X, Y, '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("Engine size")
mplt.pyplot.ylabel("Emission")
pyplot.show()

#predict the emission value
test_x_poly = poly_reg.fit_transform(test_x)
predicted_value = linear_reg.predict(test_x_poly)

#MSE and Root mean square
mse = mean_squared_error(test_y,predicted_value)
print("Mean square error",mse)
print("Root mean square error",np.sqrt(mse))

print("Mean absolute error: %.2f" % np.mean(np.absolute(predicted_value - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((predicted_value - test_y) ** 2))
print("R2-score: %.2f" % r2_score(predicted_value , test_y) )

