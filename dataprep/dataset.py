import torch 
import json
import os
import open3d as o3d
import numpy as np
import sys
from torch.utils.data import Dataset

TRAIN_TEST_SPLIT_FILE = "train_test_split_autoencoder.json"

class PointCloudDataset(Dataset):
    """ Point Cloud Dataset """

    def __init__(self, iDataset_path, iNumPoints, iSplit):
        
        self.npoints = iNumPoints

        #open the train_test_split.json
        with open (os.path.join(iDataset_path, TRAIN_TEST_SPLIT_FILE), 'r') as iFile:
            train_test_split = json.load(iFile)

        self.dataset = list()
        if iSplit == "train":
            self.dataset = train_test_split["train"]
        elif iSplit == "test":
            self.dataset = train_test_split["test"]


    def __len__(self):
        return len(self.dataset)

        
    def __getitem__(self, index):
        
        pcdFilePath = self.dataset[index]

        cloud = o3d.io.read_point_cloud(pcdFilePath)
        pointSet = np.asarray(cloud.points)

        # Should be in different transform function
        # extract only "N" number of point from the Point Cloud
        choice = np.random.choice(len(pointSet), self.npoints, replace=True)
        pointSet = pointSet[choice, :]

        # should be in different transform function
        # Normalize and center and bring it to unit sphere
        pointSet = pointSet - np.expand_dims(np.mean(pointSet, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(pointSet ** 2, axis = 1)),0)
        pointSet = pointSet / dist #scale

        # convert to pytorch tensor
        pointSet = torch.from_numpy(pointSet).float()   #convert to float32

        return pointSet


if __name__ == "__main__":
    datasetPath = r"F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\dataset_PCD_5000"
    nPoints = 4000
    split = 'train'

    d = PointCloudDataset(datasetPath, nPoints, 'train')
    print(len(d))
    points = d[11]
    print(points.shape)
    print(points)