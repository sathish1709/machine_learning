# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mplt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("Social_Network_Ads.csv")

print(df.describe())
print(df.head())

print(df.dtypes)


X = df.iloc[:, [2,3]].values
y = df.iloc[:, 4].values

#splitting training data and test data using model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)

#Feature scaling using package
from sklearn.preprocessing import StandardScaler
scalar_x = StandardScaler()
X_train = scalar_x.fit_transform(X_train)
X_test = scalar_x.fit_transform(X_test)
print("Feature scaling using Standard scaler ",X_train)

#Fitting the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train, y_train)

# make prediction
y_pred = classifier.predict(X_test)
print("Predicted value ", y_pred)

#Evaluate the model: Making confuaion matrix
from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(y_test, y_pred )
print("Confusion matrix ", cfm)

#Visualizing the values
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
mplt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
mplt.xlim(X1.min(), X1.max())
mplt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mplt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
mplt.title('Logistic Regression (Train set)')
mplt.xlabel('Age')
mplt.ylabel('Estimated Salary')
mplt.legend()
mplt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
mplt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
mplt.xlim(X1.min(), X1.max())
mplt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    mplt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
mplt.title('Logistic Regression (Test set)')
mplt.xlabel('Age')
mplt.ylabel('Estimated Salary')
mplt.legend()
mplt.show()

# This is a linear classifier but the independent variables are not linear.