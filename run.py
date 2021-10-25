from tqdm import tqdm

import imagehash

from IPython.display import HTML
import logging
import sys



import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


import json
import os
from PIL import Image
import random

import json 
import os
import time

import numpy as np
import torch

import gc
import pandas as pd
import matplotlib.pyplot as plt

from utils.get_video import get_video_youtube
from utils.evaluation import elbow_evaluation,silhouette_evaluation
from utils.clustering import run_PCA, run_clustering_algorithm
from utils.detect_face import find_all_person_in_video
from utils.facenet import run
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Initialisation DONE')
sys.path.append('./utils/yolov5')


#------- YOUTUBE VIDEO
url = 'https://www.youtube.com/watch?v=h4s0llOpKrU&ab_channel=ChristianDior'
get_video_youtube(url)


#----------- RUN GET FACES
video_path = 'MISS DIOR – The new Eau de Parfum.mp4'
aligned, names = find_all_person_in_video(video_path)

gc.collect()
torch.cuda.empty_cache()

X_pca,y,PCA_model = run_PCA(aligned,names, plot=True)
elbow_evaluation(X_pca)
silhouette_evaluation(X_pca)
number_of_clusters = 1
kmeans_model = run_clustering_algorithm(X_pca,y, number_of_clusters)

color=[]

for i in range(0,number_of_clusters+1):
    r = lambda: random.randint(0,255)
    color.append((r(),r(),r()))

#------ GET HUMAN

#@title get humans in video and render video in "results"

# @TODO: Create folder results
run(source='MISS DIOR – The new Eau de Parfum.mp4', classes=0) #Classes 0 = person
#run(source='/content/1.png', classes=0) #Classes 0 = person
