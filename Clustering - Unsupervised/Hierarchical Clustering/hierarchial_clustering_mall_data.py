import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#read data
cust_df = pd.read_csv("Mall_Customers.csv")
print(cust_df.describe())
print(cust_df.head())

#ingonre the 1st column and prepare X with
x = cust_df.iloc[:,[3,4]].values

#Using dendrogram to find the optimal cluster
#Linkage is algorithm of hirearchial clustering
#Ward = minimise the variance within cluster.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method= 'ward')) #similar to k-means "k-means++" to minimise the sum of squares within cluster.
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance between two customers to show dissimilarity")
plt.show()

#fitting the dataset to hirearchial clustering
#affinity = distance to do linkage between data points
from sklearn.cluster import AgglomerativeClustering
clusters = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_cluster = clusters.fit_predict(x)

#Visualize the cluster
#x axis is age , y axis is income
#s : scalar or array_like, shape (n, ), optional
plt.scatter(x[y_cluster ==0, 0],x[y_cluster == 0, 1 ] , s= 100, c="red", label = "Cluster1")
plt.scatter(x[y_cluster ==1, 0],x[y_cluster == 1, 1 ] , s= 100, c="blue", label = "Cluster2")
plt.scatter(x[y_cluster ==2, 0],x[y_cluster == 2, 1 ] , s= 100, c="orange", label = "Cluster3")
plt.scatter(x[y_cluster ==3, 0],x[y_cluster == 3, 1 ] , s= 100, c="yellow", label = "Cluster4")
plt.scatter(x[y_cluster ==4, 0],x[y_cluster == 4, 1 ] , s= 100, c="brown", label = "Cluster4")
plt.title("Hirearchial cluserting")
plt.xlabel('Annual Income', fontsize=18)
plt.ylabel('Spending score', fontsize=16)
plt.legend()
plt.show()
