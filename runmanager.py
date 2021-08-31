# Run Manger to manage
# 1. Every Epoch
# 2. Experiment with Runs - to be coded  

import torch
import time


class RunManager:
    def __init__(self, network, loader, summaryWriter):
        self.epoch_id  = -1
        self.epoch_loss = 0
        self.best_epoch_loss = 1e20
        self.best_epoch_id = -1
        self.epoch_start_time = None

        self.network = network
        self.loader = loader
        self.tb = summaryWriter
        
        #self.best_network = None

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_id += 1
        self.epoch_loss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time

        self.epoch_loss = self.epoch_loss / (len(self.loader.dataset) / self.loader.batch_size)
        print(f"Epoch: {self.epoch_id} Loss: {self.epoch_loss}")

        # save model with the best loss
        if self.epoch_loss < self.best_epoch_loss:
            self.best_epoch_loss = self.epoch_loss
            self.best_epoch_id = self.epoch_id
            #self.best_network = self.network.state_dict()

            torch.save(self.network.state_dict(), 'saved_models/autoencoder_%d.pth' % (self.best_epoch_id))

        # Tensorboard logging 
        # add graph for loss 
        self.tb.add_scalar('Training Loss', self.epoch_loss, self.epoch_id)  

        # add graph for accuracy

        # add historgram for weights and biases
        for name, param in self.network.named_parameters():
            if('bn' not in name and 'stn' not in name):
                self.tb.add_histogram(name, param, self.epoch_id)
                if param.grad is not None:
                    self.tb.add_histogram(name + "_grad", param.grad, self.epoch_id)


    def track_loss(self, loss):
        self.epoch_loss += loss.item()

    def being_batch():
        pass

    def end_batch():
        pass 