"""As from https://www.section.io/engineering-education/hierarchical-clustering-in-python/"""

# IMPORTS
import numpy as np # to handle numeric data
import matplotlib.pyplot as plt # for visualization
import pandas as pd # for handling dataframe

from sklearn.cluster import DBSCAN # Density-Based Spatial Clustering of Applications with Noise
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch # importing scipy.cluster.hierarchy for dendrogram

def frontier_clustering(data, algo="AGNES", metric=None, save_freq=None, num_ep_steps=1000):

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
    y_hc = clusters.fit_predict(data)  # model fitting on the dataset

    # plotting
    if save_freq and (num_ep_steps % save_freq == 0):
        plot_num = 1
        cluster_save_path = f"clustering_output/clusterplot_{plot_num}"
        visualize_clustering(data, y_hc, cluster_save_path)
        if algo_type == 'hierarchical' and dendrogram:
            dendro_save_path = f"clustering_output/dendroplot_{plot_num}"
            hierarchical_dendrogram(data, linkage, dendro_save_path)
        plot_num += 1

    means = get_frontier_cluster_region_means(data, y_hc, n_clusters)

    return means


def get_frontier_cluster_region_means(data, y_hc, n_clusters):

    # full_map = torch.zeros(num_scenes, 5, full_w, full_h).float().to(device)
    # newData = ourData.iloc[:, [6, 7]].values # extract the two features from our dataset (for plots)
    print(f"frontier map channel is of type: {type(data)}") # torch.Tensor or np.ndarray
    data_out = np.zeros(data.size())
    assert n_clusters == len(y_hc)
    # for each cluster, return the coordinates of the mean of all frontier coordinates in that cluster
    for clstr in range(n_clusters):
        cluster_coords_x, cluster_coords_y = data[y_hc == clstr, 0], data[y_hc == clstr, 1]
        cluster_mean_x, cluster_mean_y = np.mean(cluster_coords_x), np.mean(cluster_coords_y)
        data_out[cluster_mean_x, cluster_mean_y] = 1.

    return data_out

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
def visualize_clustering(data, y_hc, n_clusters, cluster_save_path):

    colours = ["red", "blue", "green", "cyan", "magenta"]
    for clstr in range(n_clusters):
        fig = plt.scatter(data[y_hc == clstr, 0], data[y_hc == clstr, 1], s=100, c=colours[clstr], label=f"cluster {clstr}")
    # plot additions
    plt.title('Frontier point clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.plot()
    plt.savefig(cluster_save_path)
    plt.close(fig)