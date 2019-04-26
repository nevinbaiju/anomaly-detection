from models.anomaly_ann import anomaly_ann
import pandas as pd
import torch
from misc.min_max import get_min_max

#from scripts.db import db
from scripts.read_video import generate_block
#from scripts.mail import send_mail

import time
import datetime
import multiprocessing
import argparse
import os

import cv2

def generate_tensor(filename):
    print(filename)
    features = pd.read_csv(filename, header=None).drop([0], axis=1).values
    features_tensor = torch.tensor(features, dtype=torch.float)
    return features_tensor

base_threshold = 0.3
alert_threshold = 0.6
verbose = True

parser = argparse.ArgumentParser()
parser.add_argument("--vid_base_path", type=str, default='SampleVideos/videos')
parser.add_argument("--features_base_path", type=str, default='SampleVideos/features')
parser.add_argument("--vid_file", type=str, default='RoadAccidents022_x264.mp4')
parser.add_argument("--weights_path", type=str, default='weights/weights_L1L2.mat')
args = parser.parse_args()

filename = args.vid_file

#db_access = db()
vid_base_path = args.vid_base_path
features_base_path = args.features_base_path
vid_file = args.vid_file
features_base_path = args.features_base_path

vid_path = os.path.join(vid_base_path, vid_file)
feature_file = vid_file.split('.')[0] + '.csv'
feature_path = os.path.join(features_base_path, feature_file)

assert (os.path.exists(vid_path)), "Video, '{}' does not exist".format(vid_path)
assert (os.path.exists(feature_path)), "Feature file, '{}' does not exist".format(feature_path)

vid = generate_block(vid_path, 1, return_frame=True)
features = generate_tensor(feature_path)
weights_path = args.weights_path

detector = anomaly_ann(weights_path, no_sigmoid=True)

cv2.namedWindow("preview")
font = cv2.FONT_HERSHEY_SIMPLEX
text_pos = (10, 30)

min, max = get_min_max(feature_file)

for i, block in enumerate(vid):
    score = detector(features[i]).item()
    score = (score-min)/(max-min)
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
