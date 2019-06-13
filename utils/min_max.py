import pandas as pd

def get_min_max(filename=''):
	if(filename.split('_')[0][:-3] == 'RoadAccidents'):
		try:
			road = pd.read_csv("/home/nevin/nevin/projects/ml/anomaly-detection/SampleVideos/features.csv", index_col='filename')
			current = road.loc[filename]
			min = current[1]
			max = current[2]
			return min, max
		except KeyError:
			return -1500, 1500
	else:
		return -1500, 1800
	return -1500, 1500
