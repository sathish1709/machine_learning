#libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#read data
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.describe())
print(cust_df.head())

#preprocessing
df = cust_df.drop('Address', axis=1)
print(df.head())

df.dropna()

#ingonre the 1st column and prepare X with
x = df.values[:,1:]
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
print(x)

#initialise k-means
#n_cluster = centroid size, n_init = number of times the algorithm runs with different centroid seeds.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(x)

#lables for each of the points
k_means_labels = k_means.labels_
print("Labels ",k_means_labels)

#centroid for each cluster
k_means_cluster_centers = k_means.cluster_centers_

df["Clus_km"] = k_means_labels
df.head(5)

area = np.pi * ( x[:, 1])**2
#x axis is age , y axis is income
#s : scalar or array_like, shape (n, ), optional
plt.scatter(x[:, 0], x[:, 3], s=area, c=k_means_labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


#3D visulaization of K means clustering for education , aga and income variable
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(x[:, 1], x[:, 0], x[:, 3], c= k_means_labels.astype(np.float))
plt.show()