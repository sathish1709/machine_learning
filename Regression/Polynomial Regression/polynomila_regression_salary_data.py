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

df = pd.read_csv("Position_Salaries.csv")

print(df.describe())
print(df.head())

print(df.dtypes)

#histogram view
#finding relationship between two columns in a dataframe
#x axis : predictive variable/independent variable
#y axis : target variable
x = np.asanyarray(df[["Level"]])
y = np.asanyarray(df[["Salary"]])
mplt.pyplot.scatter(x,y)
mplt.pyplot.xlabel("Level")
mplt.pyplot.ylabel("Salary")
mplt.pyplot.title("Relationship between level and salary Scatter Plot")
pyplot.show()

# no train test split is needed since the data contains only 10 observations. Splitting into train test will affect the accuracy.

#Now follow simple linear regression process
linear_reg = linear_model.LinearRegression()
linear_reg.fit (x, y)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

#Plot the linear regression model
mplt.pyplot.scatter(x, y,  color='blue')
mplt.pyplot.plot(x, linear_reg.predict(x), '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("Level")
mplt.pyplot.ylabel("Salary")
pyplot.show()

#if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2
poly_reg = PolynomialFeatures(degree=2)

#**fit_transform** takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
linear_reg2 = linear_model.LinearRegression()
linear_reg2.fit(x_poly,y)
predicted_value = linear_reg2.predict(x_poly)

mplt.pyplot.scatter(x, y,  color='blue')
x_bin = np.arange(min(x), max(x), 0.1)
x_bin = x_bin.reshape((len(x_bin),1))
mplt.pyplot.scatter(x, y,  color='blue')
#Y = linear_reg.intercept_[0]+ linear_reg.coef_[0][1]*X+ linear_reg.coef_[0][2]*np.power(X, 2) #Polynomial equation
# to fine Y (or) linear_reg.predict(x_poly)
mplt.pyplot.plot(x_bin, linear_reg2.predict(poly_reg.fit_transform(x_bin)), '-r') #using Y = Ɵ0 + Ɵ1 X1 + Ɵ1 X1 ^ 2
mplt.pyplot.xlabel("Levels")
mplt.pyplot.ylabel("Salary")
pyplot.show()

#MSE and Root mean square
mse = mean_squared_error(y,predicted_value)
print("Mean square error",mse)
print("Root mean square error",np.sqrt(mse))

print("Mean absolute error: %.2f" % np.mean(np.absolute(predicted_value - y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((predicted_value - y) ** 2))
print("R2-score: %.2f" % r2_score(predicted_value , y) )

