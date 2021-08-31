import torch 
import json
import os
import open3d as o3d
import numpy as np
import sys
from torch.utils.data import Dataset
from util import pointutil

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

        pointSet = pointutil.random_n_points(pointSet, self.npoints)
        pointSet = pointutil.normalize(pointSet)

        # convert to pytorch tensor
        pointSet = torch.from_numpy(pointSet).float()   #convert to float32

        pathComps = os.path.normpath(pcdFilePath).split(os.sep)[-2:]
        trimPath = os.sep.join(pathComps)

        return pointSet, trimPath


if __name__ == "__main__":
    datasetPath = r"F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\dataset_PCD_5000"
    nPoints = 4000
    split = 'train'

    d = PointCloudDataset(datasetPath, nPoints, 'train')
    print(len(d))
    points = d[11]
    print(points.shape)
    print(points)