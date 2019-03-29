from models.anomaly_detector import anomaly_detector

#from scripts.db import db
from scripts.read_video import generate_block
from scripts.mail import send_mail

import time
import datetime
import multiprocessing

def process_score(score):

    timestamp = time.time()
    ts_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(score)
    if(score>base_threshold):
        #db_access.push((ts_str, score))
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

filename = 'SampleVideos/videos/RoadAccidents022_x264.mp4'

#db_access = db()
vid = generate_block(filename, 1)
detector = anomaly_detector(weights_dict)

for i, block in enumerate(vid):
    score = detector.predict(block)
    process_score(score)
