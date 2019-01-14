
# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import linear_model
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error

df = pd.read_csv("50_Startups.csv")

print(df.describe())
print(df.head())
print(df.dtypes)

#analysis: negative linear relationship individually since the slope of the line is steep. x increases and y dccreases
#All are having a positive correlation
sb.regplot(x='R&D Spend',y='Profit', data= df)
mplt.pyplot.title("Relationship between Profit and R&D Spend: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()
print("Correlation value for R & D and Profit ",df[["R&D Spend","Profit"]].corr())

sb.regplot(x='Administration',y='Profit', data= df)
mplt.pyplot.title("Relationship between Profit and Administration: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()
print("Correlation value for Administration and Profit ",df[["Administration","Profit"]].corr())

sb.regplot(x='Marketing Spend',y='Profit', data= df)
mplt.pyplot.title("Relationship between Profit and Marketing Spend: Scatter Plot with regression line")
pyplot.ylim(0,)
pyplot.show()
print("Correlation value for Marketing and Profit ",df[["Marketing Spend","Profit"]].corr())

X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

#Specify the index of categorical column
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
print(X)

#Avoiding dummy variable trap
X = X [:, 1:]
print("X value ", X)
#splitting training data and test data using model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)


#Multiple linear regression
linear_reg = linear_model.LinearRegression()
linear_reg.fit (X_train, y_train)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

#Ordinary least square method
y_hat= linear_reg.predict(X_test)
print("Real value ", y_test)
print("Predicted value ",y_hat)

#building optimal model using backward elimination
#Formula for multiple linear reg y = b0 + b1x1 + b2x2 +...
# bo = b0x0 where x0 =1 , so we are adding a column with 1's in x value
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1)
print("X value after adding x0 column ", X)

X_optimal = X[:,[0,1,2,3,4,5]]
# Step 1: significance value set to 0.05

# Step 2: Fit the full model to possible predictor value using ordinary least sqyare regressor
OLS_reg = sm.OLS(endog = y, exog = X_optimal).fit()

# Using OLS.summary obtain P value to get the highest P-value and compare with signficance value
# If p < s then the model is complete else remove the column with heighest p value and iterate
print("Summary of OLS ",OLS_reg.summary())

# Update the x_optimal by remving column with p-value highest i.e 2
X_optimal = X[:,[0,1,3,4,5]]

# Step 2: Fit the full model to possible predictor value using ordinary least sqyare regressor
OLS_reg = sm.OLS(endog = y, exog = X_optimal).fit()

# Using OLS.summary obtain P value to get the highest P-value and compare with signficance value
# If p < s then the model is complete else remove the column with heighest p value and iterate
print("Summary of OLS ",OLS_reg.summary())

# Update the x_optimal by remving column with p-value highest i.e 1
X_optimal = X[:,[0,3,4,5]]

# Step 2: Fit the full model to possible predictor value using ordinary least sqyare regressor
OLS_reg = sm.OLS(endog = y, exog = X_optimal).fit()

# Using OLS.summary obtain P value to get the highest P-value and compare with signficance value
# If p < s then the model is complete else remove the column with heighest p value and iterate
print("Summary of OLS ",OLS_reg.summary())


# Update the x_optimal by remving column with p-value highest i.e 1
X_optimal = X[:,[0,3,5]]

# Step 2: Fit the full model to possible predictor value using ordinary least sqyare regressor
OLS_reg = sm.OLS(endog = y, exog = X_optimal).fit()


# Form the new x array with 0,3 column. 5th column is removed isnce it has higher value
X_updated = X [:, [0,3]]
print("X value updated ", X_updated)
#splitting training data and test data using model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_updated, y, test_size =0.2, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)


#Multiple linear regression
linear_reg = linear_model.LinearRegression()
linear_reg.fit (X_train, y_train)
# The coefficients
print ('Coefficients: ', linear_reg.coef_)
print ('Intercept: ',linear_reg.intercept_)

#Ordinary least square method
y_hat= linear_reg.predict(X_test)
print("Real value ", y_test)
print("Predicted value ",y_hat)

#API to iterate through backward elimiation to avoid retundant code.
s1 = 0.05
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linear_reg.score(X_test, y_test))

