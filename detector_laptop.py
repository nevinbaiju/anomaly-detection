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
parser.add_argument("--vid_base_path", type=str, default='/media/nevin/01D247D89A1BD8F0/dataset/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly')
parser.add_argument("--features_base_path", type=str, default='/home/nevin/nevin/datasets/anomaly detection/features/c3d/Training_Normal_Videos_Anomaly')
parser.add_argument("--vid_file", type=str, default='RoadAccidents022_x264')
parser.add_argument("--weights_path", type=str, default='weights/weights_L1L2.mat')
args = parser.parse_args()

filename = args.vid_file

#db_access = db()
vid_base_path = args.vid_base_path
features_base_path = args.features_base_path
vid_file = args.vid_file
features_base_path = args.features_base_path

vid_path = os.path.join(vid_base_path, vid_file+'.mp4')
feature_path = os.path.join(features_base_path, vid_file+'.csv')

assert (os.path.exists(vid_path)), "Video, '{}' does not exist".format(vid_path)
assert (os.path.exists(feature_path)), "Feature file, '{}' does not exist".format(feature_path)

vid = generate_block(os.path.join(vid_base_path, vid_file+'.mp4'), 1, return_frame=True)
features = generate_tensor(os.path.join(features_base_path, vid_file+'.csv'))
weights_path = args.weights_path

detector = anomaly_ann(weights_path, no_sigmoid=True)

cv2.namedWindow("preview")
font = cv2.FONT_HERSHEY_SIMPLEX
text_pos = (10, 30)

min, max = get_min_max(vid_file+'.csv')

for i, block in enumerate(vid):
    score = detector(features[i]).item()
    score = (score-min)/(max-min)
    if(score < 0):
        score = -score
    elif(score>1):
        score = 1
    score = score*100
    print(score)
    preview = block['preview']
    for frame in preview:
        cv2.putText(frame, "%d percent"%score, text_pos, font, 1, (255, 255, 255))
        cv2.imshow('preview', frame)
        key = cv2.waitKey(20)
        if(key==27):
            break
    #process_score(score)
