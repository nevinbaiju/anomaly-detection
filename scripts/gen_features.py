import torch
from models.C3D_features import C3D_features
from models.anomaly_ann import anomaly_ann
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scripts.read_video import generate_block
import time
from __future__ import print_function
import os



def find_length(filename, seg_length=1):
    return sum(1 for _ in generate_block(filename, 1))


def generate_c3d_features(C3D_features, filename):
    total_length = find_length(filename)
    block = generate_block(filename, 1)
    start_time = time.time()
    feature_arr = []
    for i, curr_block in enumerate(block):
        print("{}/{}".format(i+1, total_length), sep='\r', end='\r')
        features = C3D_features(curr_block)
        features = (features - features.mean())/(features.max() - features.mean())
        feature_arr.append(features.detach().numpy())
        del features
    print("")
    total_time = time.time()-start_time
    return np.rollaxis(np.array(feature_arr), 1 )[0]


def iterate_folder(base_path, net):
    start_time = time.time()
    total_files = len(os.listdir(base_path))
    category = base_path.split('/')[-1]
    print("Calculating features for :", category)

    for i, vid_file in enumerate(os.listdir(base_path)):
        features = generate_c3d_features(net, os.path.join(base_path, vid_file))
        mult = len(features)//32
        feature_norm_arr = []
        for i in range(32):
            sub_arr = np.linalg.norm(features[i*mult:(i+1)*mult], axis=0)
            feature_norm_arr.append(sub_arr)
        folder_path = os.path.join("results", "features", category)
        file_path = os.path.join(folder_path, vid_file.split('.')[0]+'.csv')
        if(os.path.exists(folder_path)):
            pd.DataFrame(np.array(feature_norm_arr, dtype='float16')).to_csv(file_path, header=False)
        else:
            os.mkdir(folder_path)
            pd.DataFrame(np.array(feature_norm_arr, dtype='float16')).to_csv(file_path, header=False)
                     
        print("{}/{} completed".format(i+1, total_files), sep='\r', end='\r')

    end_time = time.time() - start_time
    print("total time taken = ", end_time)



net = C3D_features('./weights/c3d.pickle').eval()

iterate_folder('SampleVideos/videos', net)




