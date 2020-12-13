import numpy as np
import argparse
import csv
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import chart_studio.plotly
import plotly.graph_objs as go
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List



### GENERATE DATA
def generate_data(num_records):
	centers = [[1, 1], [-1, -1], [1, -1],[-0.25, 0],[-1.5, 1]]
	X, labels_true = make_blobs(n_samples=num_records, centers=centers, cluster_std=0.4, random_state=0)
	X = StandardScaler().fit_transform(X)
	return X, labels_true




### REMOVE NOISE USES DBSCAN
def remove_noises_uses_dbscan(X):

    db = DBSCAN(eps=0.3, min_samples=10).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # Remove noise from dataset
    after_remove_noises_dataset = []
    list_noises = []
    for i in range(len(X)):
        if (labels[i] != -1):
            after_remove_noises_dataset.append(X[i])
        else:
            list_noises.append(X[i])
    
    return after_remove_noises_dataset, list_noises




### SHOW RESULT
def show_results(data: np.ndarray, labels: np.ndarray, plot_handler = "seaborn", columns_name=["x values", "y values", "Label"]) -> None:
	labels = np.reshape(labels, (1, labels.size))
	data = np.concatenate((data, labels.T), axis=1)
	
	# Seaborn plot
	if plot_handler == "seaborn":
		facet = sns.lmplot(
			data=pd.DataFrame(data, columns=columns_name), 
			x=columns_name[0], 
			y=columns_name[1], 
			hue=columns_name[2], 
			fit_reg=False, 
			legend=True, 
			legend_out=True
		)
	plt.show()





### MAIN
data, labels_true = generate_data(300) 

# show data with true label
show_results(data, labels_true)


data_after_remove_noises, list_noises = remove_noises_uses_dbscan(data)

# data = data_after_remove_noises

print("Noises: ")
print(list_noises)

print("--->Computing clusters ... ")
birch = Birch(
    branching_factor=50,
    n_clusters=5,
    threshold=0.6,
    copy=True,
    compute_labels=True
)
birch.fit(data)
predictions = np.array(birch.predict(data))



# Number of clusters in labels, ignoring noise if present.
labels = birch.labels_



# Adjusted Rand Index
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))



# DRAW RESULT
show_results(data, labels)
