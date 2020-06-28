'''
This Example sends harcoded data to Ubidots using the request HTTP
library.

Please install the library using pip install requests

Made by Jose García @https://github.com/jotathebest/
'''

import requests
import random
import time

'''
global variables
'''

ENDPOINT = "things.ubidots.com"
DEVICE_LABEL = "flaskServer"
VARIABLE_LABEL = "posture-value"
TOKEN = "BBFF-BjBE8UN70vxuFasG3L0PVLIxv0SNg6"
DELAY = 1  # Delay in seconds


def post_var(payload, url=ENDPOINT, device=DEVICE_LABEL, token=TOKEN):
    try:
        url = "http://{}/api/v1.6/devices/{}".format(url, device)
        headers = {"X-Auth-Token": token, "Content-Type": "application/json"}

        attempts = 0
        status_code = 400

        while status_code >= 400 and attempts < 5:
            print("[INFO] Sending data, attempt number: {}".format(attempts))
            req = requests.post(url=url, headers=headers,
                                json=payload)
            status_code = req.status_code
            attempts += 1
            time.sleep(1)

        print("[INFO] Results:")
        print(req.text)
    except Exception as e:
        print("[ERROR] Error posting, details: {}".format(e))


def get_posture(activity,confidence):
    tmp=activity[1: int(len(activity)-1)]
    activity_list = list(tmp.split(", ")) 

    tmp2=confidence[1: int(len(confidence)-1)]
    confidence_list = list(tmp2.split(", ")) 

    #map the right activity to a number 0- standing 1- resting 2-moving 3-eating
    posture_index =-1
    mx_confidence=0
    for i in range (len(activity_list)):
        if float(activity_list[i]) > 0 and mx_confidence < float(confidence_list[i]): 
            mx_confidence = float(confidence_list[i])
            posture_index=i
    
    return posture_index

# def main():
#     # Simulates sensor values
#     sensor_value = 6

#     # Builds Payload and topíc
#     payload = {VARIABLE_LABEL: sensor_value}

#     # Sends data
#     post_var(payload)


if __name__ == "__main__":
    while True:
        main()
        time.sleep(DELAY)