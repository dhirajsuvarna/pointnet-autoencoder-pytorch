import torch 
import argparse
import os
import random
from torch.utils.data import DataLoader
from torch import optim

from dataprep.dataset import PointCloudDataset
from model.model import PCAutoEncoder
from model.model_fxia22 import PointNetAE
from torch.utils.tensorboard import SummaryWriter
import runmanager

"""
Run this "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
for solving the error in loading the chamfer distance
"""

if torch.cuda.is_available():
    from chamfer_distance.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
else:
    from chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

#########################################################################
# SHOULD BE STRICTLY REFACTORED - this is not acceptable here
def trimfilenames(iFileName):
    # "F:\projects\ai\pointnet\dataset\DMUNet_OBJ_format\dataset_PCD_5000\Switch\Switch_4.pcd" -> "Switch\Switch_4.pcd"
    pathComps = os.path.normpath(iFileName).split(os.sep)[-2:]
    trimPath = os.sep.join(pathComps)
    return trimPath

#########################################################################

def create_network(iModelType, iNumPoints, iModelPath = ''):
    point_dim = 3
    if iModelType == 'dhiraj':
        autoencoder = PCAutoEncoder(point_dim, iNumPoints)
    elif iModelType == 'fxia':
        autoencoder = PointNetAE(iNumPoints)

    if iModelPath != '':
        autoencoder.load_state_dict(torch.load(ip_options.load_saved_model))

    # It is recommented to move the model to GPU before constructing optimizers for it. 
    # This link discusses this point in detail - https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
    # Moving the Network model to GPU
    autoencoder.to(device)
    return autoencoder


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
parser.add_argument("--num_points", type=int, required=True, help="Number of Points to sample")
parser.add_argument("--num_workers", type=int, default=4, help="Number Multiprocessing Workers")
parser.add_argument("--dataset_path", required=True, help="Path to Dataset")
parser.add_argument("--nepoch", type=int, required=True, help="Number of Epochs to train for")
parser.add_argument("--load_saved_model", default='', help="load an saved model")
parser.add_argument("--start_epoch_from", default=0, help="usually used with load model")
parser.add_argument("--model_type", required=True, choices=['dhiraj', 'fxia'], help="Model Types")

ip_options = parser.parse_args()
print(f"Input Arguments : {ip_options}")

# Seed the Randomness
manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed) #later: 

# Create instance of SummaryWriter 
writer = SummaryWriter('runs/' + ip_options.model_type)

# determine the device to run the network on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Creating Dataset
train_ds = PointCloudDataset(ip_options.dataset_path, ip_options.num_points, 'train')
test_ds = PointCloudDataset(ip_options.dataset_path, ip_options.num_points, 'test')

# Creating DataLoader 
train_dl = DataLoader(train_ds, batch_size=ip_options.batch_size, shuffle=True, num_workers= ip_options.num_workers)
test_dl = DataLoader(test_ds, batch_size=ip_options.batch_size, shuffle=True, num_workers= ip_options.num_workers)

# Output of the dataloader is a tensor reprsenting
# [batch_size, num_channels, height, width]

# getting one data sample
sample, files = next(iter(train_ds))

# Creating Model
num_points = ip_options.num_points
point_dim = 3

if ip_options.model_type == 'dhiraj':
    autoencoder = PCAutoEncoder(point_dim, num_points)
elif ip_options.model_type == 'fxia':
    autoencoder = PointNetAE(num_points)

if ip_options.load_saved_model != '':
    autoencoder.load_state_dict(torch.load(ip_options.load_saved_model))

#writer.add_graph(autoencoder, torch.tensor([28, 28]))

# iterating the weights for each layer 
for name, param in autoencoder.named_parameters():
    print(f"{name}:\t\t{param.shape}")

# It is recommented to move the model to GPU before constructing optimizers for it. 
# This link discusses this point in detail - https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
# Moving the Network model to GPU
autoencoder.to(device)



# Setting up Optimizer - https://pytorch.org/docs/stable/optim.html 
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# create folder for trained models to be saved
os.makedirs('saved_models', exist_ok=True)

# create instance of Chamfer Distance Loss Instance
chamfer_dist = ChamferDistance()

##########################################################################################
# START TRAINING OF THE NETWORK
print("Start Training...")
m = runmanager.RunManager(autoencoder, train_dl, writer)
for epoch in range(int(ip_options.start_epoch_from), ip_options.nepoch):
    m.begin_epoch()
    latent_vector_all = torch.Tensor().to(device)
    filename_all = list()
    for i, data in enumerate(train_dl):
        points = data[0]
        filenames = list(data[1])

        points = points.transpose(2, 1)
        
        points = points.to(device)

        optimizer.zero_grad()   # Reseting the gradients

        reconstructed_points, latent_vector = autoencoder(points) # perform training
        latent_vector_all = torch.cat((latent_vector_all, latent_vector), 0) 
        filename_all.extend(filenames)

        points = points.transpose(1,2)
        reconstructed_points = reconstructed_points.transpose(1,2)
        dist1, dist2 = chamfer_dist(points, reconstructed_points)   # calculate loss
        train_loss = (torch.mean(dist1)) + (torch.mean(dist2))

        print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {train_loss}")
        
        train_loss.backward() # Calculate the gradients using Back Propogation

        optimizer.step() # Update the weights and biases 

        m.track_loss(train_loss)
        # train_loss += train_loss

        # add tensorboard logging
        # for name, param in autoencoder.named_parameters():
        #     writer.add_histogram(name + "_grad", param.grad, i)

    # epoch_loss = train_loss / (len(train_ds)/ip_options.batch_size)
    # print(f"Mean Training Loss (per epoch) : {epoch_loss}")

    scheduler.step()
    m.end_epoch()

    # # save model with the best loss
    # if epoch_loss < best_loss:  
    #     best_loss = epoch_loss
    #     torch.save(autoencoder.state_dict(), 'saved_models/autoencoder_%d.pth' % (epoch))
    #     writer.add_embedding(latent_vector_all, metadata=filename_all, global_step=epoch, tag="Latent_Vectors")

    # # Tensorboard logging 
    # # 1. graph of loss function 
    # writer.add_scalar('Training Loss', epoch_loss, epoch)   

    # # 2. add historgram for weights and biases
    # for name, param in autoencoder.named_parameters():
    #     if('bn' not in name and 'stn' not in name):
    #         writer.add_histogram(name, param, epoch)
    #         if param.grad is not None:
    #             writer.add_histogram(name + "_grad", param.grad, epoch)

##########################################################################################
# find the best performing epoch and run the network to get latent vectors
print("Generate the Best Latent Vectors")
with torch.no_grad():
    best_latent_vector = torch.Tensor().to(device)
    best_filenames = list()
    autoencoder_eval = create_network(ip_options.model_type, ip_options.num_points)
    state_dict = torch.load('saved_models/network_%d.pth' %m.best_epoch_id, map_location=device)
    autoencoder_eval.load_state_dict(state_dict)
    autoencoder_eval.eval() # set the network in evaluation mode
    for itrid, data in enumerate(train_dl):
        print(f"Evaluating Batch: {itrid}")
        filenames = list(data[1])

        points = data[0]
        points = points.transpose(2, 1)        
        points = points.to(device)

        reconstructed_points, latent_vector = autoencoder_eval(points) # perform training
        best_latent_vector = torch.cat((best_latent_vector, latent_vector), 0) 
        best_filenames.extend(filenames)

    # add embedding for t-sne visualiztion
    trimmedFiles = list(map(trimfilenames, best_filenames))
    writer.add_embedding(best_latent_vector, metadata=trimmedFiles, global_step=m.best_epoch_id, tag="Latent_Vectors")

    # serialize the best latent vector
    torch.save(best_latent_vector.detach(), 'saved_models/best_latent_vector_%d.pth' %m.best_epoch_id)
    torch.save(best_filenames, 'saved_models/best_filenames_%d.pth' %m.best_epoch_id)
    best_latent_vector = best_latent_vector.cpu().data.detach().numpy()
