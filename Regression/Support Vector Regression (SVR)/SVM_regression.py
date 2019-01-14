# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("Position_Salaries.csv")

print(df.describe())
print(df.head())

print(df.dtypes)


X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
y = y.reshape(1,-1)
#splitting training data and test data using model selection
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)"""

#Feature scaling using package
from sklearn.preprocessing import StandardScaler
scalar_x = StandardScaler()
scalar_y = StandardScaler()
X = scalar_x.fit_transform(X)
print("Feature scaling using Standard scaler ",X)
y = scalar_y.fit_transform(y)
print("Feature scaling using Standard scaler ",y)

# fit the model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# make prediction
y_predict = scalar_y.inverse_transform(regressor.predict(scalar_x.transform(np.array([[7.5]]))))

# plot the graph
mplt.pyplot.scatter(X, y,  color='blue')
mplt.pyplot.plot(X, regressor.predict(X), '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("YearsExperience")
mplt.pyplot.ylabel("Salary")
pyplot.show()

# plot the graph in higher resolution
x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
mplt.pyplot.scatter(X, y,  color='blue')
mplt.pyplot.plot(x_grid, regressor.predict(x_grid), '-r') #using Y = Ɵ1 + Ɵ0 X
mplt.pyplot.xlabel("YearsExperience")
mplt.pyplot.ylabel("Salary")
pyplot.show()
