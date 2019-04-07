from models.anomaly_detector import anomaly_detector

from scripts.db import db
from scripts.read_video import generate_block
from scripts.mail import send_mail

import time
import datetime
import multiprocessing
import argparse

import cv2

def process_score(score):

    timestamp = time.time()
    ts_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(score)
    if(score>base_threshold):
        db_access.push((ts_str, score))
        if(score>alert_threshold):
            emailer = multiprocessing.Process(target=send_mail, \
                                              args=("vision.ai.updates@gmail.com", \
                                              "nevinbaiju@gmail.com", score, ts_str))
            emailer.start()
        if(verbose):
            print("Attention! Notification send to user!!")


base_threshold = 0.3
alert_threshold = 0.6
weights_dict = {'c3d' : 'weights/c3d.pickle', 'ann': 'weights/weights_L1L2.mat'}
verbose = True

parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, default='/home/nevin/priyanka.mp4')
args = parser.parse_args()

filename = args.vid

db_access = db()
vid = generate_block(filename, 1, return_frame=True)

detector = anomaly_detector(weights_dict)

cv2.namedWindow("preview")
font = cv2.FONT_HERSHEY_SIMPLEX
text_pos = (10, 30)

for i, block in enumerate(vid):
    score = detector.predict(block['block'])*100
    preview = block['preview']
    for frame in preview:
        cv2.putText(frame, "%d percent"%score, text_pos, font, 1, (255, 255, 255))
        cv2.imshow('preview', frame)
        key = cv2.waitKey(20)
        if(key==27):
            break
    #process_score(score)
