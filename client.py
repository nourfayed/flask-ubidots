import cv2
import glob
import os
import time
import numpy as np

import requests
import json
import base64
import toubidots

def encode_img(image):
    _, buffer = cv2.imencode('.jpg', image)
    enc_buff = base64.b64encode(buffer)
    return str(enc_buff, 'utf-8')

def visualize_animal_activity(img):
    url = "http://0.0.0.0:5000/model/api/v1.0/"
    headers = {'Content-Type': 'application/json'}
    image_req = json.dumps({'img': str(encode_img(img))})
    response = requests.request("POST", url=url+'recognize', headers=headers, data=image_req)
    activity = json.loads(response.content)['activity']
    confidence = json.loads(response.content)['confidence']
    return activity, confidence    

if __name__ == '__main__':

    img_dir = "/home/nourfayed/Flask_Server/dataset" # Enter Directory of all images 
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)

    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        activity, confidence = visualize_animal_activity(img)  
        #print(activity, confidence) 

        posture_index = toubidots.get_posture(activity,confidence)
        payload = {"posture-value": posture_index}
        print(f1,"  ",posture_index)
        toubidots.post_var(payload)
        time.sleep(1) # delay 1 second for the sake of the simulation 


