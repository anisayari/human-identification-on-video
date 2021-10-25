from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
from utils.detect_face import init_mtcnn,get_face
import plotly.express as px

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Initialisation DONE')

def get_embeddings_resnet(aligned):
    logging.info('Getting embeddings from Resnet...')
    from facenet_pytorch import InceptionResnetV1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.5, device=device).eval()
    embeddings_list = []
    for i in range(0, len(aligned), 100):
        stack = torch.stack(aligned[i:i+100]).to(device)
        embeddings = resnet(stack).detach().cpu()
        #print(embeddings.size())
        embeddings_list.append(embeddings)
    #stack = torch.stack(aligned[i:i+10]).to(device)
    #embeddings = resnet(stack).detach().cpu()
    #embeddings_list.append(embeddings)
    #print(np.shape(aligned))
    #aligned= np.expand_dims(aligned, axis=0)
    logging.info('Getting embeddings from Resnet DONE')
    #print(embeddings)
    #print(type(embeddings))
    stacked_embeddings = torch.cat(embeddings_list)
    #print(stacked_embeddings)
    #print(type(stacked_embeddings))
    return stacked_embeddings

def run_PCA(aligned,names, plot=True):
    logging.info('Running PCA...')


    embeddings = get_embeddings_resnet(aligned)

    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=names, index=names))

    inds = range(88)
    PCA_model = PCA(n_components=3, random_state=33).fit(embeddings)
    X_pca1 = PCA_model.transform(embeddings)
    y = [i for i in range(0,len(names))]
    if plot:
        total_var = PCA_model.explained_variance_ratio_.sum() * 100

        fig = px.scatter_3d(
            X_pca1, x=0, y=1, z=2, color=y,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
        fig.show()

        exp_var_cumul = np.cumsum(PCA_model.explained_variance_ratio_)

        fig = px.area(
            x=range(1, exp_var_cumul.shape[0] + 1),
            y=exp_var_cumul,
            labels={"x": "# Components", "y": "Explained Variance"}
        )
        fig.show()
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        img = ax.scatter(X_pca1[:, 0], X_pca1[:, 1], c=y, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
            
        plt.title('PCA method')
        plt.suptitle('Face embeddings')

        cbar = plt.colorbar(img, ax=ax)
        #cbar.ax.set_yticklabels(np.unique(trainLabels[inds])) 
        plt.show()
        """
    logging.info('Running PCA DONE')
    return X_pca1,y,PCA_model

def run_clustering_algorithm(X_pca1,y,number_of_clusters):
    logging.info('Running clustering algorithm...')
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(X_pca1)
    y_kmeans = kmeans.predict(X_pca1)

    plt.scatter(X_pca1[:, 0], X_pca1[:, 1], c=y_kmeans, s=50, cmap='viridis')

    fig = px.scatter_3d(
            X_pca1, x=0, y=1, z=2, color=y_kmeans,
            title=f'Kmeans',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    fig.show()

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    logging.info('Running clustering algorithm DONE')
    return kmeans

def get_class_face(frame, kmeans_model, PCA_model):
    mtcnn = init_mtcnn()
    img_cropped, prob = get_face(frame,mtcnn)
    id = [-1]
    if img_cropped is not None:
        embeddings = get_embeddings_resnet([img_cropped])
        X = PCA_model.transform(embeddings)
        id = kmeans_model.predict(X)
    return id[0]

