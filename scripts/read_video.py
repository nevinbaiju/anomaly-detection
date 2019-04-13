import torch
import cv2
from misc.mean import get_mean, get_std
from torchvision.transforms import Normalize, Compose, Resize, ToPILImage, ToTensor
import numpy as np

def get_tensor(arr, norm_parameters):
	"""
	Function to convert the given block of frames to tensor of required shape (ch, fr, h, w)

	Parameters
	----------
	arr       :list
			   List of frames in the current block of frames.
	segment_length  :int
			 The length of the given segment as multipliers of 16.

	Returns
	-------
	weights:     torch.tensor
				 Tensor of the block in the required shape.
	"""
	transforms = Compose([\
							  ToPILImage(),\
							  Resize((112, 112), interpolation=2),\
							  ToTensor(),\
							  Normalize(mean=norm_parameters['mean'], std=norm_parameters['std'])\
						 ])
	blocc = [transforms(img).view(1, 3, 112, 112) for img in arr]
	blocc = torch.cat(blocc)
	blocc = torch.transpose(blocc, 0, 1)
	blocc = blocc.view(1, 3, 16, 112, 112)

	return blocc
	
def generate_block(video, segment_length, return_frame=False):
	"""
	Function to generate the video segments from the given file.

	Parameters
	----------
	arr            :str
					Path of the video file
	segment_length :int
					The length of the given segment as multipliers of 16.

	Yields
	-------
	Video Segments :torch.tensor
					Tensor of the block in the required shape.
	"""
	cap = cv2.VideoCapture(video)
	# Check if camera opened successfully
	i = 0
	arr = []
	frame_counter = 0
	curr_frame = []
	norm_parameters = {'mean': get_mean(), 'std': get_std()}
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		#print(frame)
		if (ret == True):
			if(i<16*segment_length):
				arr.append(frame) #cv2.resize(frame, (112, 112), interpolation = cv2.INTER_AREA)
				i+=1
				frame_counter +=1
			else:
				i = 0
				arr = []

			if(len(arr) == (16*segment_length)):
				X = get_tensor_numpy(arr, norm_parameters)
				if(return_frame):
					yield {'preview': arr, 'block': X}
				else:
					yield {'block': X}
		else:
			cap.release()
			return

if __name__ == '__main__':
	block = generate_block('../SampleVideos/videos/RoadAccidents022_x264.mp4', 3)
	for i, curr_block in enumerate(block):
		print(i)
