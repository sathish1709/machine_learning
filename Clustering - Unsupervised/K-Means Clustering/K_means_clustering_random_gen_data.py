#libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#setting previous number using seed function
print(np.random.seed(0))

#x: Array of shape [n_samples, n_features]. (Feature Matrix)
#The generated samples.
#y: Array of shape [n_samples]. (Response Vector)
#The integer labels for cluster membership of each sample.
x, y = make_blobs(n_samples=5000, centers=[[8,8], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
print("X value",x)
print("Y value",y)

#X axis: all rows of 0th column
#Y axis: all rows of 1th column
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.show()

#initialise k-means
#n_cluster = centroid size, n_init = number of times the algorithm runs with different centroid seeds.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(x)

#lables for each of the points
k_means_labels = k_means.labels_
print("Labels ",k_means_labels)

#centroid for each cluster
k_means_cluster_centers = k_means.cluster_centers_
print("Centroid ",k_means_cluster_centers)

#plot graph
# Create a plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[8, 8], [-2, -1], [2, -3], [1, 1]])), colors):
    # Create a list of all data points, where the data poitns that are
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans clustering')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()