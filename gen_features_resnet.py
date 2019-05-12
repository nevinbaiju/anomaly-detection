# This script iterates over the video files in a folder and writes them in a csv file.

from __future__ import print_function

from models.ResNet_3D.model import generate_model
from models.ResNet_3D.opts import parse_opts
from utils.mean import get_mean, get_std

import torch

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.read_video import generate_block
import time
import os
import argparse



def find_length(filename, seg_length=1):
    """
    Function to find the length of the video in terms of specified segments.
    """
    return sum(1 for _ in generate_block(filename, 1))

def get_file_list(filename):
    """
    Function to get the files in the file list as a list.
    """
    with open(filename, 'r') as file:
        f = file.read()
    return f.split('\n')[:-1]

def get_norm_features(features):
    """
    Function to get the normalized features
    """
    mult = len(features)//32
    feature_norm_arr = []
    for i in range(32):
        sub_arr = np.linalg.norm(features[i*mult:(i+1)*mult], axis=0)
        feature_norm_arr.append(sub_arr)
    return np.array(feature_norm_arr, dtype='float16')

def save_features(features, category, vid_file):
    """
    Function to save the features as csv
    """
    folder_path = os.path.join(output_dir, category)
    file_path = os.path.join(folder_path, vid_file.split('.')[0]+'.csv')
    if not(os.path.exists(folder_path)):
        os.mkdirs(folder_path)
    pd.DataFrame(features, index=None).to_csv(file_path, header=False)

def generate_resnet_features(resnet, filename):
    """
    Function to process the current video file and generate the resnet fetures.

    Parameters
    resnet          :torch.nn.Module
                     The resnet feature extraction model.
    filename        :str
                     The filename of the videos to return the features

    Returns
    -------
    features        :numpy.array
                     The features of the video in the required shape.
    """
    total_length = find_length(filename)
    block = generate_block(filename)
    start_time = time.time()
    feature_arr = []
    for i, curr_block in enumerate(block):
        print("\t\t\t\t\t [{}/{}]".format(i+1, total_length), sep='\r', end='\r')
        features = resnet(curr_block['block'])
        #features = (features - features.mean())/(features.max() - features.mean())
        feature_arr.append(features.cpu().detach().numpy())
        del features
    total_time = time.time()-start_time
    features = np.rollaxis(np.array(feature_arr), 1 )[0]
    return features

def iterate_list(file_list, base_path, resnet):
    """
    Function to iterate over the video files in the given file list
    and call the function to generate the resnet features and write the
    features in a csv file.
    Parameters
    ----------
    file_list       :str
                     filename of the file containing the list of videos.
    base_path       :str
                     Path of the folder containing the video files.
    resnet          :torch.nn.Module
                     The resnet feature extraction model.
    Returns
    -------
    None
    """
    start_time = time.time()
    category = base_path.split('/')[-1]
    total_files = len(file_list) - 1
    print("Calculating features for :", category)

    for i, vid_file in enumerate(file_list):
        print("Processing [{}/{}]".format(i, total_files), sep='\r', end='\r')
        vid_file = vid_file.split('/')[-1]
        features = generate_resnet_features(resnet, os.path.join(base_path, vid_file))

        if not(no_norm):
            features = get_norm_features(features)
        save_features(features, category, vid_file)

    end_time = time.time() - start_time
    print("total time taken = ", end_time)

def iterate_folder(base_path, resnet):
    """
    Function to iterate over the video files in the given folder
    and call the function to generate the resnet features and write the
    features in a csv file.
    Parameters
    ----------
    base_path       :str
                     Path of the folder containing the video files.
    resnet          :torch.nn.Module
                     The resnet feature extraction model.
    Returns
    -------
    None
    """
    start_time = time.time()
    total_files = len(os.listdir(base_path))
    category = base_path.split('/')[-1]
    print("Calculating features for :", category)

    for i, vid_file in enumerate(os.listdir(base_path)):
        print("Processing [{}/{}]".format(i, total_files), sep='\r', end='\r')
        features = generate_resnet_features(resnet, os.path.join(base_path, vid_file))
        print(features.shape, vid_file)

        if not(no_norm):
            features = get_norm_features(features)
        save_features(features, category, vid_file)

    end_time = time.time() - start_time
    print("total time taken = ", end_time)

opt = parse_opts()

opt.mean = get_mean()
opt.std = get_std()
opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
opt.sample_size = 112
opt.sample_duration = 16
opt.n_classes = 400

base_path = opt.base_path
file_list_mode = opt.file_list_mode
file_list = opt.file_list
no_cuda = opt.no_cuda
no_norm = opt.no_norm
output_dir = args.output_dir

if (no_cuda):
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

model = generate_model(opt)
print('loading model {}....'.format(opt.model))
model_data = torch.load(opt.model)
assert opt.arch == model_data['arch']
model.load_state_dict(model_data['state_dict'],strict=False)
model = model.eval()
print("Done!")

if(file_list_mode):
    assert (os.path.exists(file_list)), "File list not found or not specified."
    vid_list = get_file_list(file_list)
    iterate_list(base_path=base_path, file_list=vid_list, resnet=model)
else:
    iterate_folder(base_path, model)
