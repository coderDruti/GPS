import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path):
    file = open(file_path, "r")
    cols = file.readline().split("\t")

    data = []
    for x in file:
        data.append(x.split("\t"))

    y = []
    for i in range(0, len(data)):
        y.append(data[i][0])
        data[i].pop(0)

    df = pd.DataFrame(data, index = y, columns = cols)
    return df

def preprocess_data(df):
    scaler = MaxAbsScaler()
    X_train_maxAbs = scaler.fit_transform(df)
    X_train_maxAbs
    pca = PCA(n_components = 2)
    pca_data_maxAbs = pca.fit_transform(X_train_maxAbs)
    # print(pca_data_maxAbs)
    # for i in range(5):
    #     print("--", sep="")
    # new_arr=[row for i,row in enumerate(pca_data_maxAbs) if pca_data_maxAbs[i][0]>0]
    # print(len(new_arr))
    return pca_data_maxAbs

def run_clustering(pca_data_maxAbs):
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i,init="random",random_state=32)
        kmeans.fit(pca_data_maxAbs)
        inertias.append(kmeans.inertia_) #append to the array, sum of squared distances of samples to their nearest cluster center

    # plt.plot(range(1,11), inertias, marker='o')
    # plt.grid()
    # plt.title('Elbow method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.show()

    kmeans = KMeans(init = "random", n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(pca_data_maxAbs)
    return clusters.tolist()


def results(clusters, pca_data_maxAbs):
    # plt.scatter(pca_data_maxAbs[:, 0], pca_data_maxAbs[:, 1], c=clusters, cmap='viridis')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA followed by KMeans Clustering')
    # plt.colorbar(label='Cluster')
    # plt.show()
    score = silhouette_score(pca_data_maxAbs, clusters)
    return score

def main(file_path):  
    df = load_data(file_path)
    pca_data_maxAbs = preprocess_data(df)
    clusters = run_clustering(pca_data_maxAbs)
    # silhouette_score = results(clusters,pca_data_maxAbs)
    df.to_csv("data.csv")
    return clusters, pca_data_maxAbs[:,0], pca_data_maxAbs[:,1]