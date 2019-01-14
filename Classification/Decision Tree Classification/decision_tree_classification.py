# data preprocessing

# loading libraries

import numpy as np
import pandas as pd
import matplotlib as mplt
from matplotlib import pyplot
import seaborn as sb
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

df = pd.read_csv("drug200.csv", delimiter=",")

print(df.describe())
print(df.head())

print(df.size)
print(df.dtypes)

#data preprocessing
x = df[["Age","Sex","BP","Cholesterol","Na_to_K"]].values


#converting categorical variable to numeric variable using labelencoder

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3])

#Another way of converting categorical to numeric
#Creating dummies for a coulumns
dummy_variables = pd.get_dummies(df['Sex'])
print(dummy_variables.head())
dummy_variables.rename(columns={'F':'Female','M':'Male'},inplace=True)

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variables], axis=1)
print(df.head())


print(x[0:5])

#target value assignment
y = df["Drug"].values
y = np.asarray(df["Drug"].values)

print("Target Value",y[0:5])


#test and train data split
x_train, y_train, x_test, y_test = train_test_split(x,y, test_size=0.3, random_state=3)
print ('Train set data:', x_train,  y_train)
print ('Test set data:', x_test ,  y_test)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


#assigning the object
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(type(x_train))
print(type(y_train))
decision_tree.fit(x_train,y_train)

predict_value = decision_tree.predict(x_test)
print ("Predicted Value",predict_value [0:5])
print ("Actual Value", y_test [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predict_value))

#Plotting graph
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(decision_tree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')




