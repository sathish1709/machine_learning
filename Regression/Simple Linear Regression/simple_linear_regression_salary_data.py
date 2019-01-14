# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
import seaborn as sb
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy import stats

df = pd.read_csv("Salary_Data.csv")
print(df.head())
print(df.dtypes)
print(df.describe())

#analysis: negative linear relationship  since the slope of the line is steep. x increases and y dccreases
sb.regplot(x='YearsExperience',y='Salary', data= df)
mplt.pyplot.title("Relationship between YearsExperience and Salary: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()

# Using pandas corr()
# Analyse correlation usinf Pearson correlation with correlation coefficient and p-value
# Correlation coefficinet: Close to +1 : Positive relationship ; close to -1: Negative relationship; 0+ no relationship
# p-value: Strong certainty : p < 0.001 , Moderate : p < 0.05 , weak : p <0.1 , No certainty : p > 0.1
print("Correlation value for highway-mpg and price",df[["YearsExperience", "Salary"]].corr())

#Using stats module
correlation_coefficinet, p_value = stats.pearsonr(df['YearsExperience'], df['Salary'])
print("correlation coefficient",correlation_coefficinet,"and p-value",p_value)

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


#splitting training data and test data using model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)

#linear regression Y = Ɵ1 + Ɵ0 X ; Y= dependent variable, X = independent variable, Ɵ0= slope, Ɵ1= intercept
#No future scaling is needed for simple linear regression, the library has its own feature scaling
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train,y_train)

# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

mplt.pyplot.scatter(X_train, y_train,  color='blue')
mplt.pyplot.plot(X_train, linear_reg.coef_* X_train + linear_reg.intercept_, '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("YearsExperience")
mplt.pyplot.ylabel("Salary")
pyplot.show()

mplt.pyplot.scatter(X_test, y_test,  color='blue')
mplt.pyplot.plot(X_train, linear_reg.coef_* X_train + linear_reg.intercept_, '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("YearsExperience")
mplt.pyplot.ylabel("Salary")
pyplot.show()

#predict price with the linear model using user input data
z = float(input("Enter the year of experience"))
#converting into 2D array
a = np.reshape(z, (1, 1))
predicted_salary = linear_reg.predict(a)
print(f"The predicted salary for {z}",predicted_salary)


# reshape
predicted_salary_val = linear_reg.predict(X_test)
print("The predicted_salary_val for the test set is ",predicted_salary_val)


#Calculate MSE (Ordinary least square method) and Root mean square error
#R2 is not an error, it is metric for accuracy of model.
#Higher the R2, better the model designed.
print("using Y = Ɵ1 + Ɵ0 X",linear_reg.coef_ *X_train + linear_reg.intercept_)
#mse = mean_squared_error(train['CO2EMISSIONS'],linear_reg.coef_[0][0]*train_x + linear_reg.intercept_[0])
mse = mean_squared_error(X_test,predicted_salary_val)
print("Mean square error",mse)
print("Root mean square error",np.sqrt(mse))
