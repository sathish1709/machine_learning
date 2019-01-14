# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
df = pd.read_csv('Data.csv')

#data preprocessing using Pandas
mean_age = df["Age"].astype(float).mean()
print(mean_age)
df["Age"].replace(np.nan, mean_age,inplace=True)
print(df["Age"].values)

mean_salary = df["Salary"].astype(float).mean()
print(mean_salary)
df["Salary"].replace(np.nan, mean_salary,inplace=True)
print(df["Salary"].values)

X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

#Creating dummies for a coulumns
dummy_variables = pd.get_dummies(df['Country'])
print(dummy_variables.head())

dummy_variables.rename(columns={'Country-France':'France','Country-Germany':'Germany','Country-Spain':'Spain'},inplace=True)

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variables], axis=1)
print(df.head())
df.drop(labels="Country", axis=1 , inplace = True)
print(df.head())

#Data preprocessing using packages like labelencoder and onehotencoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

#y = labelencoder_y.fit_transform(y)
# Alternate using package labelencoder and onehotencoder
# Encoding categorical data
# Encoding the Independent Variable
# Use labelEncoder to convert categorial data to numeric value
# Onehotencoder to create dummy variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)

#Specify the index of categorical column
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)

#splitting data in test and train using pandas dataframe
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
train_x = np.asanyarray(train[['France','Germany','Spain','Age','Salary']])
print("Train x using pandas",train_x)
train_y = np.asanyarray(train[['Purchased']])
print("Train y using pandas", train_y)
test_x = np.asanyarray(test[['France','Germany','Spain','Age','Salary']])
print("Train x using pandas",test_x)
test_y = np.asanyarray(test[['Purchased']])
print("Train y using pandas", test_y)


#splitting training data and test data using model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)
print("X train",X_train)
print("X test",X_test)
print("y train",y_train)
print("y test",y_test)

#feature Scaling: Particularly done when there is no standardisation in data.
#In our dataset, if we compute age and salary column eucledian distance, it will show huge difference.
#Feature scaling should be done to prodive uniformity or normalisation.
#Normalisation
# #Simple feature Scaling using numpy
train_x= train_x/train_x.max()
print(train_x)

test_x= test_x/test_x.max()
print(test_x)

# #Simple feature Scaling using pandas dataframe
df["Salary"]= df["Salary"]/df["Salary"].max()
print("Simple feature scaling: ",df["Salary"].values)

#Min- Max method
df["Salary"]= (df["Salary"]-df["Salary"].min())/(df["Salary"].min()-df["Salary"].max())
print("Min max scaling: ",df["Salary"].values)

#z test method
df["Salary"]= (df["Salary"]-df["Salary"].mean())/(df["Salary"].std())
print("z test scaling method: ",df["Salary"].values)

#Feature scaling using package
from sklearn.preprocessing import StandardScaler
scalar_x = StandardScaler()
X_train = scalar_x.fit_transform(X_train)
print("Feature scaling using Standard scaler ",X_train)
X_test = scalar_x.fit_transform(X_test)
print("Feature scaling using Standard scaler ",X_test)