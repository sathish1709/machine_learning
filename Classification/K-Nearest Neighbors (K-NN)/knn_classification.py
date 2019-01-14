# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

df = pd.read_csv("teleCust1000t.csv")

print(df.describe())
print(df.head())
print(df['custcat'].value_counts())

print(df.dtypes)

df.hist(column='income', bins=50)

#fetching 5 rows of the below column
x = df [['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
print(x[0:5])

y = df[['custcat']].values
print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print ('Train set data:', x_train,  y_train)
print ('Test set data:', x_test ,  y_test)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

k = 4
#Train Model and Predict
neighbor = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
print("Neighbor",neighbor)

yhat = neighbor.predict(x_test)
print(yhat[0:5])

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neighbor.predict(x_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

cm = confusion_matrix(y_test, yhat)
print(cm)

k_value = 10
mean_acc = np.zeros((k_value - 1))
std_acc = np.zeros((k_value - 1))

for n in range(1, k_value):
    # Train Model and Predict
    print("Value of k ", n)
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, neighbor.predict(x_train)))
    yhat = neigh.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    print("Mean accuracy of test data ", mean_acc[n - 1])
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
    print("Standard accuracy of test data ", std_acc[n - 1])
    cm = confusion_matrix(y_test, yhat)
    print("Confusion martix", cm)

#find the best K value using maximum mean accuracy
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

#Plot a graph
mplt.pyplot.plot(range(1,k_value),mean_acc,'g')
mplt.pyplot.legend(('Accuracy ', '+/- 3xstd'))
mplt.pyplot.ylabel('Accuracy ')
mplt.pyplot.xlabel('Number of Neighbors (K)')
mplt.pyplot.tight_layout()
mplt.pyplot.show()
