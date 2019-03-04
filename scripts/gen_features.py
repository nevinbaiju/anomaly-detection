# This script iterates over the video files in a folder and writes them in a csv file.

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
    """
    Function to find the length of the video in terms of specified segments.
    """
    return sum(1 for _ in generate_block(filename, 1))


def generate_c3d_features(c3d, filename):
    """
    Function to process the current video file and generate the c3d fetures.
    
    Parameters
    c3d             :torch.nn.Module
                     The c3d feature extraction model.
    filename        :str
                     The filename of the videos to return the features
    
    Returns
    -------
    features        :numpy.array
                     The features of the video in the required shape.
    """
    total_length = find_length(filename)
    block = generate_block(filename, 1)
    start_time = time.time()
    feature_arr = []
    for i, curr_block in enumerate(block):
        print("{}/{}".format(i+1, total_length), sep='\r', end='\r')
        features = c3d(curr_block)
        features = (features - features.mean())/(features.max() - features.mean())
        feature_arr.append(features.detach().numpy())
        del features
    print("")
    total_time = time.time()-start_time
    features = np.rollaxis(np.array(feature_arr), 1 )[0]
    return features


def iterate_folder(base_path, c3d):
    """
    Function to iterate over the video files in the given folder
    and call the function to generate the c3d features and write the
    features in a csv file.
    Parameters
    ----------
    base_path       :str
                     Path of the folder containing the video files.
    c3d             :torch.nn.Module
                     The c3d feature extraction model.
    Returns
    -------
    None
    """
    start_time = time.time()
    total_files = len(os.listdir(base_path))
    category = base_path.split('/')[-1]
    print("Calculating features for :", category)

    for i, vid_file in enumerate(os.listdir(base_path)):
        features = generate_c3d_features(c3d, os.path.join(base_path, vid_file))
        mult = len(features)//32
        feature_norm_arr = []
        for i in range(32):
            sub_arr = np.linalg.norm(features[i*mult:(i+1)*mult], axis=0)
            feature_norm_arr.append(sub_arr)
        folder_path = os.path.join("results", "features", category)
        file_path = os.path.join(folder_path, vid_file.split('.')[0]+'.csv')
        if not(os.path.exists(folder_path)):
            os.mkdir(folder_path)
        pd.DataFrame(np.array(feature_norm_arr, dtype='float16')).to_csv(file_path, header=False)
                     
        print("{}/{} completed".format(i+1, total_files), sep='\r', end='\r')

    end_time = time.time() - start_time
    print("total time taken = ", end_time)

parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default='../SampleVideos/videos')
parser.add_argument("--c3d_weights", type=str, default='../weights/c3d.pickle')
args = parser.parse_args()

base_path = args.base_path
c3d_weights = args.c3d_weights

net = C3D_features(c3d_weights).eval()

iterate_folder(base_path, net)




