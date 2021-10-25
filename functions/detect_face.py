import torch
import logging
import cv2
from tqdm import tqdm

def init_mtcnn():
    logging.info('Init face recognition model...')
    from facenet_pytorch import MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(device=device)
    print('Running on device: {}'.format(device))
    logging.info('Init face recognition model DONE')
    return mtcnn

def get_face(frame,mtcnn):
    #save_path = f'/content/results/faces_found/frame{number_of_frame}-output.jpg'
    img_cropped,prob = mtcnn(frame, save_path=None,return_prob=True)
    return img_cropped, prob

def find_all_person_in_video(video_path):
    # Create a VideoCapture object
    # If required, create a face detection pipeline using MTCNN:
    # Create an inception resnet (in eval mode):

    mtcnn = init_mtcnn()
    #img = Image.open('/content/1.png').convert('RGB')

    aligned = []
    names = []

    cap = cv2.VideoCapture(video_path)
    total_number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for number_of_frame in tqdm(range(0,total_number_of_frame)):
        if number_of_frame > 200: #for debug purpose
            break
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #number_of_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        img_cropped, prob = get_face(frame,mtcnn)

        if img_cropped is not None:
            #print(prob)
            #print('Face detected with probability: {:8f}'.format(prob))
            aligned.append(img_cropped)
            names.append(f"frame{number_of_frame}")

        #cv2.imwrite('/content/results/face_found/results.jpg',frame)
    return aligned, names
