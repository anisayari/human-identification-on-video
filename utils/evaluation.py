import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Initialisation DONE')

#ELBOW EVALUATION ---------- optional 
def elbow_evaluation(X_pca1):
    logging.info('Running elbow evaluation...')
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X_pca1)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    logging.info('Running elbow evaluation DONE')

def silhouette_evaluation(X_pca):
    logging.info('Running silhouette evaluation...')


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
    plt.title('Silhouette score')
    plt.show()
    logging.info('Silhouette evaluation DONE')
