import numpy as np # to handle numeric data
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling dataframe

# READING IN AND PREPPING DATA
ourData = pd.read_csv('Pokemon.csv') # read the data # 'Mall_Customers.csv'
ourData.head() # print the first five rows of our dataset

newData = ourData.iloc[:, [6, 7]].values # extract the two features from our dataset

print(newData)

# EXECUTING CLUSTERING ON DATA WITH CHOICE OF N-CLUSTERS
from sklearn.cluster import DBSCAN # Density-Based Spatial Clustering of Applications with Noise

Density = DBSCAN(eps=3)
y_hc = Density.fit_predict(newData) # model fitting on the dataset

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