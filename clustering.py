"""As from https://www.section.io/engineering-education/hierarchical-clustering-in-python/"""

# IMPORTS
import numpy as np # to handle numeric data
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling dataframe

from sklearn.cluster import DBSCAN # Density-Based Spatial Clustering of Applications with Noise
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch # importing scipy.cluster.hierarchy for dendrogram

def frontier_clustering(data, step, data_form="map", algo="AGNES", metric=None, save_freq=None):

    # initialization
    defaults = {"AGNES": {"type": 'hierarchical', "metric": 10, "n_clusters": 5, "linkage": 'ward', "dendrogram": None},
                "DIANA": {"type": 'hierarchical', "metric": 10, "n_clusters": 5, "linkage": 'ward', "dendrogram": None},
                "DBSCAN": {"type": 'density', "metric": 3.5}
                }
    assert algo in defaults.keys()
    if not metric:
        metric = defaults[algo]["metric"]

    algo_type = defaults[algo]["type"]
    if algo_type == 'hierarchical':
        algo_info = defaults[algo]
        n_clusters, linkage, dendrogram = algo_info["n_clusters"], algo_info["linkage"], algo_info["dendrogram"]

    # clustering
    if algo == "DBSCAN":
        clusters = DBSCAN(eps=metric)
    elif algo == "AGNES":
        clusters = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage)

    print(f"frontier map channel is of type: {type(data)}, and of data-form: {data_form}")  # torch.Tensor or np.ndarray
    print(f"DATA_IN_SHAPE: {data.shape}")
    if data_form == "map":
        num_rows, num_cols = np.shape(data)[0], np.shape(data)[1]
        data = map_to_columns(data, num_rows)
    elif data_form == "columns":
        num_rows = num_cols = np.amax(data)
    print(f"DATA_MID_SHAPE: {data.shape}")
    y_hc = clusters.fit_predict(data)  # model fitting on the dataset
    print(f"DATA_CLUSTERING_SHAPE: {y_hc}")
    if y_hc.shape[0] == 0:
        save_freq = 1
    # calculating cluster means
    means_col = get_frontier_cluster_region_means(data, y_hc, n_clusters)
    print(means_col)
    means_map = columns_to_map(means_col, num_rows, num_cols)
	
    print(f"DATA_OUT_SHAPE: {means_map.shape}")
    print(f"MEANS_MAP: {np.amax(means_map)}")
    
    # plotting
    if save_freq and (step % save_freq == 0):
        plot_num = 1
        cluster_save_path = f"clustering_output/clusterplot_{algo}_{plot_num}"
        n_clusters = max(y_hc)+1
        visualize_clustering(data, y_hc, means_col, n_clusters, cluster_save_path)
        if algo_type == 'hierarchical' and dendrogram:
            dendro_save_path = f"clustering_output/dendroplot_{algo}_{plot_num}"
            hierarchical_dendrogram(data, linkage, dendro_save_path)
        plot_num += 1
	
    return means_map


def map_to_columns(data, num_rows):
    nzero_is = np.nonzero(data)
    nzero_is_sw = np.transpose(nzero_is)
    columns = np.flip(nzero_is_sw, axis=1)
    slice = columns[:,1]
    convert_slice = [(-1*x)-1+num_rows for x in slice]
    columns[:,1] = convert_slice
    return columns

def columns_to_map(data, r, c):
    map_out = np.zeros([r, c])
    for i, row in enumerate(data):
        map_out[r-1-int(row[1]), int(row[0])] = 1.
    return map_out

def get_frontier_cluster_region_means(data, y_hc, n_clusters):

    # full_map = torch.zeros(num_scenes, 5, full_w, full_h).float().to(device)
    # newData = ourData.iloc[:, [6, 7]].values # extract the two features from our dataset (for plots)
    # for each cluster, return the coordinates of the mean of all frontier coordinates in that cluster
    column = np.zeros([n_clusters, 2])
    for clstr in range(n_clusters):
        cluster_coords_x, cluster_coords_y = data[y_hc == clstr, 0], data[y_hc == clstr, 1]
        cluster_mean_x, cluster_mean_y = np.mean(cluster_coords_x), np.mean(cluster_coords_y)
        cluster_mean_x, cluster_mean_y = int(round(cluster_mean_x,0)), int(round(cluster_mean_y,0))
        column[clstr][0], column[clstr][1] = cluster_mean_x, cluster_mean_y

    return column

# plotting dendrogram, also allows finding the optimal number of clusters
def hierarchical_dendrogram(data, linkage, dendro_save_path):

    fig = sch.dendrogram(sch.linkage(data, method=linkage))
    plt.title('Dendrogram')
    plt.xlabel('Frontier points')
    plt.ylabel('Euclidean distances')
    plt.plot()
    plt.savefig(dendro_save_path)
    plt.close(fig)

# VISUALIZING THE CLUSTERING
def visualize_clustering(data, y_hc, means_col, n_clusters, cluster_save_path):

    colours = ["red", "blue", "green", "cyan", "magenta", "yellow"]
    for clstr in range(n_clusters):
        plt.scatter(data[y_hc == clstr, 0], data[y_hc == clstr, 1], s=100, c=colours[clstr], label=f"cluster {clstr}")
    plt.scatter(means_col[:, 0], means_col[:, 1], s=100, c=colours[clstr+1], label=f"cluster means")
    # plot additions
    plt.title('Frontier point clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.plot()
    plt.savefig(cluster_save_path)
    plt.close()

if __name__ == "__main__":
    # ourData = pd.read_csv('Pokemon.csv')
    ourData = pd.read_csv('Mall_Customers.csv')
    ourData.head()  # print the first five rows of our dataset
    data = ourData.iloc[:, [3, 4]].values  # extract the two features from our dataset
    step = 0

    frontier_clustering(data, step, data_form="columns", algo="AGNES", metric=None, save_freq=1)
