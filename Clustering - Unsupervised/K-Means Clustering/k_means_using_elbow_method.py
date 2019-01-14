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

#finding optimal K value using elbow method
#n_cluster = centroid size, n_init = number of times the algorithm runs with different centroid seeds.
# k-means++  to avoid random initialization trap, max_iter = maximum number of cluster
wcss = []
for i in range(1,11):
    k_means = KMeans(init = "k-means++", n_clusters = i, max_iter=300, n_init = 10, random_state= 0)
    k_means.fit(x)
    #compute within cluster sum of squares
    wcss.append(k_means.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("number of cluster")
plt.ylabel("WCSS")
plt.show()

#nOW THE k VALUE = 5 , after performing elbow method.
k_means = KMeans(init = "k-means++", n_clusters = 5, max_iter=300, n_init = 10, random_state= 0)
y_kmeans = k_means.fit_predict(x)

#x axis is age , y axis is income
#s : scalar or array_like, shape (n, ), optional
plt.scatter(x[y_kmeans ==0, 0],x[y_kmeans == 0, 1 ] , s= 100, c="red", label = "Cluster1")
plt.scatter(x[y_kmeans ==1, 0],x[y_kmeans == 1, 1 ] , s= 100, c="blue", label = "Cluster2")
plt.scatter(x[y_kmeans ==2, 0],x[y_kmeans == 2, 1 ] , s= 100, c="orange", label = "Cluster3")
plt.scatter(x[y_kmeans ==3, 0],x[y_kmeans == 3, 1 ] , s= 100, c="yellow", label = "Cluster4")
plt.scatter(x[y_kmeans ==4, 0],x[y_kmeans == 4, 1 ] , s= 100, c="brown", label = "Cluster4")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1], s= 300, c="magenta", label = "Centroid")
plt.title("k means cluserting")
plt.xlabel('Annual Income', fontsize=18)
plt.ylabel('Spending score', fontsize=16)
plt.legend()
plt.show()


