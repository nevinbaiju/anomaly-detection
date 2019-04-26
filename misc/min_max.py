import pandas as pd

def get_min_max(filename=''):
    if(filename.split('_')[0][:-3] == 'RoadAccidents'):
        road = pd.read_csv("/home/nevin/RoadAccidents.csv", index_col='filename')
        current = road.loc[filename]
        min = current.min_val
        max = current.max_val
        return min, max
    else:
        return -1800, 2000
