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

if torch.cuda.is_available():
    from chamfer_distance.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
else:
    from chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

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
sample = next(iter(train_ds))

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

# Start the Training 
best_loss = 1e20
for epoch in range(int(ip_options.start_epoch_from), ip_options.nepoch):
    for i, data in enumerate(train_dl):
        points = data
        points = points.transpose(2, 1)
        
        points = points.to(device)

        optimizer.zero_grad()   # Reseting the gradients

        reconstructed_points = autoencoder(points) # perform training

        points = points.transpose(1,2)
        reconstructed_points = reconstructed_points.transpose(1,2)
        dist1, dist2 = chamfer_dist(points, reconstructed_points)   # calculate loss
        train_loss = (torch.mean(dist1)) + (torch.mean(dist2))

        print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {train_loss}")
        
        train_loss.backward() # Calculate the gradients using Back Propogation

        optimizer.step() # Update the weights and biases 
        train_loss += train_loss

        # add tensorboard logging
        # for name, param in autoencoder.named_parameters():
        #     writer.add_histogram(name + "_grad", param.grad, i)

    epoch_loss = train_loss / (len(train_ds)/ip_options.batch_size)
    print(f"Mean Training Loss (per epoch) : {epoch_loss}")

    scheduler.step()

    # save model with the best loss
    if epoch_loss < best_loss:  
        best_loss = epoch_loss
        torch.save(autoencoder.state_dict(), 'saved_models/autoencoder_%d.pth' % (epoch))

    # Tensorboard logging 
    # 1. graph of loss function 
    writer.add_scalar('Training Loss', epoch_loss, epoch)   

    # 2. add historgram for weights and biases
    for name, param in autoencoder.named_parameters():
        if('bn' not in name and 'stn' not in name):
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(name + "_grad", param.grad, epoch)

        


        



