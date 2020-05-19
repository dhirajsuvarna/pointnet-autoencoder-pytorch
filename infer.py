import torch
import argparse
import os
from model.model import PCAutoEncoder
import open3d as o3d
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input_folder", required=True, help="Single 3d model or input folder containing 3d models")
parser.add_argument("--nn_model", required=True, help="Trained Neural Network Model")

ip_options = parser.parse_args()
input_folder = ip_options.input_folder


device = torch.device('cpu')
state_dict = torch.load(ip_options.nn_model, map_location=device)

point_dim = 3
num_points = 4000

autoencoder = PCAutoEncoder(point_dim, num_points)
autoencoder.load_state_dict(state_dict)


def infer_model_file(input_file, autoencoder):
    
    cloud = o3d.io.read_point_cloud(input_file)
    points = np.array(cloud.points)

    # extract only "N" number of point from the Point Cloud
    choice = np.random.choice(len(points), num_points, replace=True)
    points = points[choice, :]

    # Normalize and center and bring it to unit sphere
    points = points - np.expand_dims(np.mean(points, axis = 0), 0) # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis = 1)),0)
    points = points / dist #scale

    points = torch.from_numpy(points).float()
    points = torch.unsqueeze(points, 0) #done to introduce batch_size of 1 
    points = points.transpose(2, 1)
    #points = points.cuda() #uncomment this if running on GPU
    autoencoder = autoencoder.eval()
    reconstructed_points, global_feat = autoencoder(points)

    #Reshape 
    reconstructed_points = reconstructed_points.squeeze().transpose(0,1)
    reconstructed_points = reconstructed_points.numpy()

    # Create pcd file of the reconstructed points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reconstructed_points)
    outPcdFile = os.path.join(os.path.dirname(input_file), os.path.splitext(os.path.basename(input_file))[0] + "_out.pcd")
    o3d.io.write_point_cloud(outPcdFile, pcd, write_ascii=True)



def infer_models_folder(input_folder, autoencoder):
    for root, subdirs, files in os.walk(ip_options.input_folder):
        for fileName in files:
            ipFilePath = os.path.join(root, fileName)
            # check the file 
            if ipFilePath.endswith('.pcd'):
                infer_model_file(ipFilePath, autoencoder)
       

with torch.no_grad():
    if os.path.isdir(input_folder):
        infer_models_folder(input_folder, autoencoder)
    elif os.path.isfile(input_folder):
        infer_model_file(input_folder, autoencoder)



