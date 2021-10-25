import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score

#ELBOW EVALUATION ---------- optional 
def elbow_evaluation(X_pca1):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X_pca1)
        distortions.append(kmeanModel.inertia_)
    print(distortions)
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('./plot/elbow_evaluation.png', dpi=200) 
    coeff= []
    for i in range(len(distortions)-1):
        coeff.append(distortions[i+1] + distortions[i-1] - 2 * distortions[i])
    best_k = coeff.index((max(coeff)))+1
    print('BEST K FOUND {}'.format(best_k))
    return best_k

def silhouette_evaluation(X_pca):
    silhouette = []
    K = range(2,10)

    for k in K:
        kmeans = KMeans(n_clusters = k).fit(X_pca)
        labels = kmeans.labels_
        silhouette.append(silhouette_score(X_pca, labels, metric = 'euclidean'))
    
    plt.figure(figsize=(16,8))
    plt.plot(K, silhouette, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.savefig('./plot/silhouette_evaluation.png', dpi=200) 
    plt.show()
