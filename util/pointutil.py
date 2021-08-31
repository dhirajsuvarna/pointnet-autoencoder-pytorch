
import numpy as np

def normalize(iPoints):
    # Normalize and center and bring it to unit sphere
    iPoints = iPoints - np.expand_dims(np.mean(iPoints, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(iPoints ** 2, axis = 1)),0)
    iPoints = iPoints / dist #scale
    return iPoints

def random_n_points(iPoints, iNum):
    # extract only "N" number of point from the Point Cloud
    choice = np.random.choice(len(iPoints), iNum, replace=True)
    iPoints = iPoints[choice, :]
    return iPoints
