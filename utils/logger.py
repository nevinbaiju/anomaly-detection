import datetime
import json

def write_log(anomaly_score):
    curr_ts = str(datetime.datetime.now())
    log_str = curr_ts + ', ' + anomaly_score
    log_str_json = json.dumps({"time_stamp" : curr_ts, "anomaly_score" : anomaly_score})
    with open("log.csv", 'a') as file:
        file.write(log_str+'\n')
    with open("log.json", 'a') as file:
        file.write(log_str_json+'\n')