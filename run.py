
from IPython.display import HTML
import logging

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

import torch
import random
import torch
import gc
import argparse

from functions.get_video import get_video_youtube
from functions.evaluation import elbow_evaluation,silhouette_evaluation
from functions.clustering import run_PCA, run_clustering_algorithm
from functions.detect_face import find_all_person_in_video
from functions.facenet import run

from pathlib import Path

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Detect person in video and track faces from youtube video')
    parser.add_argument('URL',
                        metavar='url',
                        type=str,
                        help='url of the youtube video')

    args = parser.parse_args()
    print(args)
    url = args.URL
    #url = 'https://www.youtube.com/watch?v=h4s0llOpKrU&ab_channel=ChristianDior'

    plot = True
    if plot:
        Path("./plot/").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename='run.log')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info('Initialisation DONE')


    #------- YOUTUBE VIDEO
    logger.info('Video downloading...')
    video_file_path = get_video_youtube(url)
    logger.info('Video downloaded')


    #----------- RUN GET FACES
    video_path = 'MISS DIOR – The new Eau de Parfum.mp4'

    logging.info('Finding all faces in the video...')
    aligned, names = find_all_person_in_video(video_file_path)
    logging.info('Finding all faces in the video DONE')

    gc.collect()
    torch.cuda.empty_cache()

    logging.info('Running PCA...')
    X_pca,y,PCA_model = run_PCA(aligned,names, plot=True)
    logging.info('Running PCA DONE')

    plot = True
    if plot:
        Path("plot/").mkdir(parents=True, exist_ok=True)

    logging.info('Running elbow evaluation...')
    best_k = elbow_evaluation(X_pca)
    logging.info('Running elbow evaluation DONE')

    logging.info('Running silhouette evaluation...')
    silhouette_evaluation(X_pca)
    logging.info('Silhouette evaluation DONE')

    number_of_clusters = best_k

    logging.info('Running clustering algorithm...')
    kmeans_model = run_clustering_algorithm(X_pca,y, number_of_clusters)
    logging.info('Running clustering algorithm DONE')

    color=[]
    for i in range(0,number_of_clusters+1):
        r = lambda: random.randint(0,255)
        color.append((r(),r(),r()))

    logging.info('Running humans detection...')
    run(kmeans_model = kmeans_model,PCA_model=PCA_model, source=video_file_path, classes=0, color= color) #Classes 0 = person
    logging.info('Running humans detection DONE')

    #run(source='/content/1.png', classes=0) #Classes 0 = person
