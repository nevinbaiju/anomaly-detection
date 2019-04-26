from models.anomaly_detector import anomaly_detector

#from scripts.db import db
from scripts.read_video import generate_block
from misc.min_max import get_min_max
#from scripts.mail import send_mail

import time
import datetime
import multiprocessing
import argparse

import cv2


base_threshold = 0.3
alert_threshold = 0.6
weights_dict = {'c3d' : 'weights/c3d.pickle', 'ann': 'weights/weights_L1L2.mat'}
verbose = True

parser = argparse.ArgumentParser()
parser.add_argument("--vid", type=str, default='SampleVideos/videos/RoadAccidents022_x264.mp4')
parser.add_argument('--no_sigmoid', action='store_true', help='If true, no sigmoid for ann')
args = parser.parse_args()

filename = args.vid
no_sigmoid = args.no_sigmoid
if filename == '0':
    filename = 0

#db_access = db()
vid = generate_block(filename, 1, return_frame=True)
csv_index = filename.split('/')[-1].split('.')[0]+'.csv'
min, max = get_min_max(csv_index)

detector = anomaly_detector(weights_dict, no_sigmoid='True')

cv2.namedWindow("preview")
font = cv2.FONT_HERSHEY_SIMPLEX
text_pos = (10, 30)

for i, block in enumerate(vid):
    score = detector.predict(block['block'])
    score = (score-min)/(max-min)
    disp_score = score*100
    if(score < 0):
        score = -score
    elif(score>1):
        score = 1
    disp_score = score*100
    print(score)
    preview = block['preview']
    for frame in preview:
        frame = cv2.resize(frame, (500, 360))
        cv2.putText(frame, "%d percent"%disp_score, text_pos, font, 1, (255, 255, 255))
        cv2.imshow('preview', frame)
        key = cv2.waitKey(20)
        if(key==27):
            break
    #process_score(score)
