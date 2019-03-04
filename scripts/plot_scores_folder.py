# This script plots the scores for the videos in a folder

import torch
from models.C3D_features import C3D_features
from models.anomaly_ann import anomaly_ann

import matplotlib.pyplot as plt
import numpy as np
from scripts.read_video import generate_block
import time
import os
import argparse

def plot_score(score_arr, base_path, filename):
	"""
	Function to plot the final scores obtained from the anomaly ann.
	
	Parameters
	----------
	score_arr       :list
					List of the anomaly scores of segments in order
	base_path       :str
					Base path of the folder in which the plot images 
					will be saved.
	filename        :str
					Filename of the plot figure.
	
	Returns
	-------
	None
	"""
	if not(os.path.exists(base_path)):
			os.mkdir(base_path)
	fig = plt.figure()
	plt.plot(score_arr)
	fig.savefig(os.path.join(base_path, filename))
	del fig
	plt.close('all')

def predict_scores(c3d, anomaly_ann, filename, seg_length, base_path):
	"""
	Function to predict the anomaly scores of a video and plot the scores
	by iterating over the segment length of the video.
	
	Parameters
	----------
	c3d             :torch.nn.Module
					 The c3d feature extraction model.
	anomaly_ann     :torch.nn.Module
					 The anomaly ann model that predicts the anomaly scores.
	filename        :str
					 Filename of the plot figure.
	seg_length      :int
					 length of the video segment to be processed as
					 multipliers of 16.
	base_path       :str
					 Path of the folder containing the video files.
	
	Returns
	-------
	None
	"""
	block = generate_block(os.path.join(base_path, filename), seg_length)
	score_arr = []
	start_time = time.time()
	for i, curr_block in enumerate(block):
		features = c3d(curr_block)
		features = features.norm(dim=0)
		features = (features-features.min())/(features.max()-features.min())
		scores = anomaly_ann(features)
		del features
		print(scores,sep='\r', end='\r')
		score_arr.append(scores.item())
		del scores
	total_time = time.time()-start_time
	plot_score(score_arr, os.path.join('results/plots', base_path.split('/')[-1]), filename.split('.')[0]+'.png')

def iterate_folder(base_path, c3d, anomaly_ann):
	"""
	Function to iterate over the video files in the given folder
	and call the function to predict the anomaly for each.
	
	Parameters
	----------
	base_path       :str
					 Path of the folder containing the video files.
	c3d             :torch.nn.Module
					 The c3d feature extraction model.
	anomaly_ann     :torch.nn.Module
					 The anomaly ann model that predicts the anomaly scores.
	
	Returns
	-------
	None
	"""
	start_time = time.time()
	total_files = len(os.listdir(base_path))
	print("Calculating scores for :", base_path.split('/')[-1])

	for i, vid_files in enumerate(os.listdir(base_path)):
		predict_scores(c3d, anomaly_ann, vid_files, 1, base_path)
		print("{}/{} completed".format(i+1, total_files), sep='\r', end='\r')

	end_time = time.time() - start_time
	print("total time taken = ", end_time)



parser = argparse.ArgumentParser()
parser.add_argument("--base_path", type=str, default='../SampleVideos/videos')
parser.add_argument("--c3d_weights", type=str, default='../weights/c3d.pickle')
parser.add_argument("--ann_weights", type=str, default='../weights/weights_L1L2.mat')
args = parser.parse_args()

base_path = args.base_path
c3d_weights = args.c3d_weights
ann_weights = args.ann_weights

c3d = C3D_features(weights_path=c3d_weights).eval()
anomaly_ann = anomaly_ann(weights_path=ann_weights).eval()


iterate_folder(base_path, c3d, anomaly_ann)