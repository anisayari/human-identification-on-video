
import logging
import pytube
#from base64 import b64encode

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Initialisation DONE')

def get_video_youtube(url):
    try:
        youtube = pytube.YouTube(url)
    except:
        logger.error(f'The URL {url} is not a correct youtube video link')

    video = youtube.streams.get_highest_resolution()
    video_file_path = video.download() # In Same Folder

    """ TO READ ON NOTEBOOK
    mp4 = open('MISS DIOR – The new Eau de Parfum.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    logger.info('Displaying the video...')

    """
    return video_file_path