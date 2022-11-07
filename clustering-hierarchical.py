"""As from https://www.section.io/engineering-education/hierarchical-clustering-in-python/"""

# IMPORTS
import numpy as np # to handle numeric data
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling dataframe

# READING IN AND PREPPING DATA
ourData = pd.read_csv('Mall_Customers.csv') # read the data
ourData.head() # print the first five rows of our dataset

newData = ourData.iloc[:, [3, 4]].values # extract the two features from our dataset

# MAKING A DENDROGRAM/EXECUTING CLUSTERING ON DATA TO CHOOSE N-CLUSTERS
import scipy.cluster.hierarchy as sch # importing scipy.cluster.hierarchy for dendrogram
dendrogram = sch.dendrogram(sch.linkage(newData, method = 'ward')) # finding the optimal number of clusters using dendrogram
plt.title('Dendrogram') # title of the dendrogram
plt.xlabel('Customers') # label of the x-axis
plt.ylabel('Euclidean distances') # label of the y-axis
plt.show() # show the dendrogram

# EXECUTING CLUSTERING ON DATA WITH CHOICE OF N-CLUSTERS
from sklearn.cluster import AgglomerativeClustering # this line of code imports AgglomerativeClustering model from sk-learn
'''
we need to create an AgglomerativeClustering object, and in it, we pass the following parameters:
n_cluster= 5, the number of clusters our model should return
affinity=euclidean, specify metric to be used to calculate distances
linkage= ward to regulate how distance calculation will be carried out between different clusters.
'''
Agg_hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = Agg_hc.fit_predict(newData) # model fitting on the dataset

# VISUALIZING THE CLUSTERING
# plotting cluster 1
plt.scatter(newData[y_hc == 0, 0], newData[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plotting cluster 2
plt.scatter(newData[y_hc == 1, 0], newData[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # plotting cluster 3
plt.scatter(newData[y_hc == 2, 0], newData[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plotting cluster 4
plt.scatter(newData[y_hc == 3, 0], newData[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  # plotting cluster 5
plt.scatter(newData[y_hc == 4, 0], newData[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plot title addition
plt.title('Clusters of customers')
# labelling the x-axis
plt.xlabel('Annual Income (k$)')
# label of the y-axis
plt.ylabel('Spending Score (1-100)')
# printing the legend
plt.legend()
# show the plot
plt.show()