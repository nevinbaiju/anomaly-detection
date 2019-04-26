from .db import db
from .mail import send_mail

import time
import datetime
import multiprocessing


def process_score(score):

    timestamp = time.time()
    db_access = db()
    verbose = True
    base_threshold, alert_threshold = 0.65, 0.75
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

if __name__ == '__main__':
    process_score(0.71)
