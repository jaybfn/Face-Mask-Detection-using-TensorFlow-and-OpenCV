import logging
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import math

def write_image(out, frame):
    """
    writes frame from the webcam as png file to disk. datetime is used as filename.
    """
    if not os.path.exists(out):
        os.makedirs(out)
    now = datetime.now() 
    dt_string = now.strftime("%H-%M-%S-%f") 
    filename = f'{out}/{dt_string}.png'
    logging.info(f'write image {filename}')
    cv2.imwrite(filename, frame)


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(1)
    if k == 113: # q button to quit the video
        return 'q'
    if k == 32: # space bar to click pictures once you are inside the rectangular box
        return 'space'
    if k == 112: # p key calculates the prediction of the input images
        return 'p'
    return None


def init_cam(width, height):
    """
    setups and creates a connection to the webcam
    """

    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise ConnectionError("Could not open video device")
    
    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    # cv2.putText()
    return NotImplementedError

def predict_frame(image,model):
    """
    1. Resizing the image from 224*224 pixel to 128*128 pixel
    2. Changing to numpy array
    3. Expanding dimensions
    4. Making predictions
    """
    image = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    incomming_image = np.array(image)/255
    incomming_image_exp = np.expand_dims(incomming_image, axis = 0)

    y_pred = model.predict(incomming_image_exp)

    if y_pred[0][0] > y_pred[0][1]: 
        print(f'NO MASK!!.. DANGER!! : prediction accuracy{round((y_pred[0][0]),2)*100}%')
    else:
        print(f'Got masked : prediction accuracy{round((y_pred[0][1]),2)*100}%')
        
    
