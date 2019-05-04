from .db import db
from .mail import send_mail

import time
import datetime
import multiprocessing

class score_processor():
    """
    Class for processing the score for logging and sending notifications.
    """
    def __init__(self, base_threshold, alert_threshold, verbose):
        self.base_threshold = base_threshold
        self.alert_threshold = alert_threshold
        self.verbose = verbose
        self.db_access = db()

    def process_score(self, score):
        timestamp = time.time()
        ts_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        if(self.verbose):
            print(score)
        if(score>self.base_threshold):
            self.db_access.push((ts_str, score))
            if(score>self.alert_threshold):
                emailer = multiprocessing.Process(target=send_mail, \
                                                  args=("vision.ai.updates@gmail.com", \
                                                  "nevinbaiju@gmail.com", score, ts_str))
                emailer.start()
                if(self.verbose):
                    print("Attention! Notification send to user!!")

if __name__ == '__main__':
    score_processor = score_processor(0.5, 0.7, True)
    score_processor.process_score(0.8)
