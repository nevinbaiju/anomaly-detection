import torch
import cv2
from skimage.transform import resize
import numpy as np

def get_tensor(arr, segment_length):
	"""
    Function to convert the given block of frames to tensor of required shape (ch, fr, h, w)

    Parameters
    ----------
    arr 			:list
         			 List of frames in the current block of frames.
    segment_length	:int
    				 The length of the given segment as multipliers of 16.

    Returns
    -------
    weights:     torch.tensor
                 Tensor of the block in the required shape.
    """
	blocc = np.array([cv2.resize(frame, (112, 112), interpolation = cv2.INTER_AREA) for frame in arr])
	#blocc = blocc[:, :, 44:44+112, :]
	blocc = blocc.transpose(3, 0, 1, 2)  # ch, fr, h, w
	#blocc = np.expand_dims(blocc, axis=0)  # batch axis
	blocc = np.array(np.split(blocc, segment_length, axis=1))
	#blocc = (blocc-blocc.mean())/(blocc.max()-blocc.mean())
	blocc = np.float32(blocc)
	blocc = torch.from_numpy(blocc)
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
	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		#print(frame)
		if (ret == True):
			if(i<16*segment_length):
				arr.append(frame)
				i+=1
				frame_counter +=1
			else:
				i = 0
				arr = []

			if(len(arr) == (16*segment_length)):
				X = get_tensor(arr, segment_length)
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
