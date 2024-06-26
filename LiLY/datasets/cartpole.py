import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import ipdb as pdb

class CartpoleDataset(Dataset):

    def __init__(self, directory, dataset='cartpole', transform='default'):
        super().__init__()
        self.path = os.path.join(directory, dataset)
        self.domain_names = ['v11']
        self.num_domains = 1 #len(self.domain_names)
        self.datums_per_domain = 1000
        self.samples_per_datum = 40
        self.length = 3

    def __len__(self):
        length = self.num_domains * self.datums_per_domain * (self.samples_per_datum - self.length + 1)
        return length
    
    def __getitem__(self, idx):
        offset = self.samples_per_datum - self.length + 1
        src_domain = idx % self.num_domains
        src_rollout = (idx // self.num_domains) // offset
        src_timestep = (idx // self.num_domains) % offset
        data_path = os.path.join(self.path, 
                                 str(self.domain_names[src_domain]), 
                                 'trail_%d.npz'%src_rollout)
        datum = np.load(data_path)
        frames = datum['obs'][src_timestep:src_timestep+self.length]
        frames = frames.reshape(self.length,1,128,128)
        # x, x_dot, theta, theta_dot = state
        states = datum['state'][src_timestep:src_timestep+self.length, [0,2]]
        # Only use the first action in the sequence
        actions = datum['action'][src_timestep]
        sample = {'xt': frames.astype('float32'), 
                  'yt': states.astype('float32'), 
                  'at': actions, 
                  'ct': src_domain}

        return sample

class CartpoleDatasetTwoSample(Dataset):

    def __init__(self, directory, dataset='cartpole', transform='default'):
        super().__init__()
        self.path = os.path.join(directory, dataset)
        self.domain_names = ['v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21']
        self.num_domains = 11 #len(self.domain_names)
        self.datums_per_domain = 1000
        self.samples_per_datum = 40
        self.length = 3

    def __len__(self):
        length = self.num_domains * self.datums_per_domain * (self.samples_per_datum - self.length + 1)
        return length
    
    def retrieve_by_index(self, idx):
        offset = self.samples_per_datum - self.length + 1
        src_domain = idx % self.num_domains
        src_rollout = (idx // self.num_domains) // offset
        src_timestep = (idx // self.num_domains) % offset
        data_path = os.path.join(self.path, 
                                 str(self.domain_names[src_domain]), 
                                 'trail_%d.npz'%src_rollout)
        datum = np.load(data_path)
        frames = datum['obs'][src_timestep:src_timestep+self.length]
        frames = frames.reshape(self.length,1,128,128)
        # x, x_dot, theta, theta_dot = state
        states = datum['state'][src_timestep:src_timestep+self.length, [0,2]]
        # Only use the first action in the sequence
        actions = datum['action'][src_timestep]
        return frames.astype('float32'), states.astype('float32'), actions, src_domain

    def __getitem__(self, idx):
        xt, yt, at, ct = self.retrieve_by_index(idx)
        idx_rnd = random.randint(0, self.__len__()-1)
        xtr, ytr, atr, ctr = self.retrieve_by_index(idx_rnd)

        sample = {"s1": {"yt": yt, "xt": xt, "ct": ct},
                  "s2": {"yt": ytr, "xt": xtr, "ct": ctr}
                  }

        return sample