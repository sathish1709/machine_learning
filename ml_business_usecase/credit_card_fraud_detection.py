#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import scipy
import seaborn as sb

#read data
df = pd.read_csv("creditcard.csv")
print(df.describe())

print(df.dtypes)

print(df.head())

print(df.columns)

print(df.shape)

#Understanding data: Class 0 = valid transaction and class 1 = Fradualent treansaction
sampled_data = df.sample(frac=0.5, random_state=1)
print(sampled_data.shape)
sampled_data.hist(figsize=(30,30))
plt.show()

fraudal_count = sampled_data[sampled_data["Class"] == 1]
legal_count = sampled_data[sampled_data["Class"] == 0]
print("Fradual transaction count :",fraudal_count)
print("Legal transaction count :",legal_count)

outlier_value = len(fraudal_count)/float(len(legal_count))
print("Outlier value ", outlier_value)

fig = plt.figure(figsize=(10,10))
sb.heatmap(data = sampled_data.corr(), square= True)
plt.title("Heat map to show relationships among each column")
plt.show()

y_value = sampled_data[["Class"]]
print(y_value.head())
y_value = y_value.values.reshape(-1)
print(y_value.shape)
x_value = sampled_data.drop(labels="Class", axis = 1)
print(x_value.columns)
print(x_value.shape)


# Print shapes
print(x_value.shape)
print(y_value.shape)

#Algorithms used: Random Isolation, LocalOutlier factor are common  anomaly detection methods
random_isolation = IsolationForest(max_samples=len(x_value), contamination = outlier_value, random_state= 3 )
local_outlier = LocalOutlierFactor(n_neighbors= 12, contamination= outlier_value)

n_outlier = len(fraudal_count)
#fit and predict
random_isolation.fit(x_value)
score_prediction = random_isolation.decision_function(x_value)
y_predict_lof = random_isolation.predict(x_value)

y_predict_isf = local_outlier.fit_predict(x_value)
score_prediction = local_outlier.negative_outlier_factor_


#Change the value to 0 for valid and 1 for fradual cases.
y_predict_isf[y_predict_isf == 1] = 0
y_predict_isf[y_predict_isf == -1] = 1
y_predict_lof[y_predict_lof == 1] = 0
y_predict_lof[y_predict_lof == -1] = 1

n_error_isf = (y_predict_isf != y_value).sum()
n_error_lof = (y_predict_lof != y_value).sum()
print("Error value for Isolation forest ",n_error_isf)
print("Error value for local outlier function ",n_error_lof)

print(accuracy_score(y_value,y_predict_isf))
print(accuracy_score(y_value,y_predict_isf))

print(classification_report(y_value,y_predict_lof))
print(classification_report(y_value,y_predict_isf))