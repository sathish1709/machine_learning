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

#Scatter plot
mplt.pyplot.scatter(x="FUELCONSUMPTION_COMB", y="CO2EMISSIONS",  color='blue', data= df)
mplt.pyplot.xlabel("FUELCONSUMPTION_COMB")
mplt.pyplot.ylabel("Emission")
mplt.pyplot.show()

#analysis of linear relationship  since the slope of the line is steep. x increases and y dccreases
sb.regplot(x='CYLINDERS',y='CO2EMISSIONS', data= df)
mplt.pyplot.title("Relationship between cylinder and emission: Scatter Plot with regression line")
pyplot.ylim(0,)
mplt.pyplot.xlabel("cylinder")
mplt.pyplot.ylabel("Emission")
pyplot.show()

#splitting data in test and train using 80-20 rule.
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#importing linear model
#Simple linear regression
linear_reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
linear_reg.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

mplt.pyplot.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
mplt.pyplot.plot(train_x, linear_reg.coef_[0][0]*train_x + linear_reg.intercept_[0], '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("Engine size")
mplt.pyplot.ylabel("Emission")
pyplot.show()

#predict price with the linear model using user input data
z = float(input("Enter the engine size"))
#converting into 2D array
a = np.reshape(z, (1, 1))
predicted_emission = linear_reg.predict(a)
print(f"The pricted price for engine size of {z}",predicted_emission)


data2list = test["CO2EMISSIONS"].index.tolist()
data2list = np.reshape(data2list,(len(data2list),1))
print("Shape of data",data2list.shape)
# reshape
predicted_emission_val = linear_reg.predict(data2list)
print("The predicted_emission_value for engine size ",predicted_emission_val)


#Calculate MSE and Root mean square error
print("using Y = Ɵ1 + Ɵ0 X",linear_reg.coef_[0][0]*train_x + linear_reg.intercept_[0])
#mse = mean_squared_error(train['CO2EMISSIONS'],linear_reg.coef_[0][0]*train_x + linear_reg.intercept_[0])
mse = mean_squared_error(test['CO2EMISSIONS'],predicted_emission_val)
print("Mean square error",mse)
print("Root mean square error",np.sqrt(mse))
