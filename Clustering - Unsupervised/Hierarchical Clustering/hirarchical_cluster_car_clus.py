import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy.cluster.hierarchy import fcluster
import pylab

df = pd.read_csv("cars_clus.csv")

print("Shape ",df.shape)
print(df.head())

#Data cleaning
print ("Shape of dataset before cleaning: ", df.size)
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", df.size)
print(df.head(5))

#Feature set
feature_set = df[['engine_s','horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap','mpg']]

#Normalisation
x = feature_set.values
clus_dataSet = MinMaxScaler().fit_transform(x)
print("Clustered Dataset ",clus_dataSet[0:5])

leng = clus_dataSet.shape[0]
print("length ",leng)
D = scipy.zeros([leng,leng])
print(D.shape)

#Two for loop initialised to find the euclidean distance for all points in the cluster.
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(clus_dataSet[i], clus_dataSet[j])
        print(f"Euclidean distance of {[i,j]} ",D[i,j])

#Distance between the cluster single, complete, average, weighted, centroid
Z = hierarchy.linkage(D, 'complete')

max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print("Clusters", clusters)

fig = pylab.figure(figsize=(18, 50))

#plot the dendrogram:
def llf(id):
    return '[%s %s %s]' % (df['manufact'][id], df['model'][id], int(float(df['type'][id])))

#dendrogram
dendro = hierarchy.dendrogram(Z, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
print("Dendrogram ",dendro)

#Forming six clusters and the labels are from 0 to 5.
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(clus_dataSet)
print("Labels ",agglom.labels_)

#adding new feild to the dataframe to show the cluster grouping
df['cluster_'] = agglom.labels_
print(df.head())

#cluster colors
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = df[df.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

df.groupby(['cluster_','type'])['cluster_'].count()
agg_cars = df.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()


plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()